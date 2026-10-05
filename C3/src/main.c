// INTUNE C3: supervisor del brazo (ESP32 aparte).
//
// Etapa 3: movimiento por pasos cortos (motion.c) hacia el RoArm-M2-S por UART (arm_uart.c; ESP-NOW
// como alternativa en menuconfig), con niveles de
// alerta y heartbeat del S3 (safety.c, arm_protocol.h). El brazo conserva su firmware de fábrica.
// Sin heartbeat del S3 nada se mueve: para probar sin S3, simularlo con "s3 <nivel>".
//
// Comandos (línea + Enter, 115200):
//   act <b> <s> <e> <h> [etiqueta]  como go, pero es una ACCIÓN: pulso de sync (GPIO, alto ~1 ms) + EVENT al S3
//                             en el primer paso. etiqueta 0 = correcta, 1 = error deliberado, 2 = sin etiquetar
//   onset [n] [grados]        prueba de inicio: n acciones (20) de la base, LED + pulso en cada inicio
//   onset stop                aborta la prueba
//   synctest                  pin de sync en alto 3 s (comprobar con multímetro)
//   led                       3 destellos del LED (¿tiene LED la placa?)
//   s3 <0-3|off>              simula el S3 (ALERT cada 100 ms por el mismo parser); off = S3 muerto
//   confirm                   = pulsación corta del botón: sale de la pausa (nivel 2) si nivel <= 1
//   reset                     = pulsación larga: rearma tras parada (nivel 3 / sin heartbeat)
//   home                      movimiento lento a la postura inicial; después se sabe dónde está el brazo
//   go <b> <s> <e> <h>        destino en rad (base, hombro, codo, pinza), enviado por pasos
//   gob <b>                   solo la base
//   vel <grados/s>            velocidad máxima de los pasos (por defecto 20)
//   stop                      frena: destino = posición actual
//   freeze / resume           prueba: deja de transmitir de golpe (supervisor "muerto") / reanuda
//   json <JSON>               JSON crudo (bloquea T:0, T:1041 y spd/acc 0); invalida la posición
//   (solo con enlace ESP-NOW:)
//   scan                      busca el AP del brazo, fija canal y peer
//   cmd <1|2>                 campo cmd del mensaje (2 = loop() del brazo, por defecto)
//   peer <ap|bcast>           unicast a la MAC del AP del brazo (con ACK, por defecto) o broadcast
//   det                       estado del detector ErrP local (CONFIG_ARM_LOCAL_DETECTOR)
//   status
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver/uart.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "actions.h"
#include "detector.h"
#include "arm_uart.h"
#include "motion.h"
#include "roarm_espnow.h"
#include "safety.h"

static const char *TAG = "sup";

static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static uint8_t s_arm_ap[6];
static bool s_arm_found;
static uint8_t s_channel;
static bool s_unicast = true;
static uint8_t s_cmd = CONFIG_ARM_ESPNOW_CMD;

static volatile int64_t s_tx_us;           // instante del último esp_now_send
static volatile bool s_tx_verbose;         // registrar la confirmación del último envío
static uint32_t s_sent, s_ok, s_fail;

__attribute__((unused)) static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    int64_t dt = esp_timer_get_time() - s_tx_us;
    bool ok = status == ESP_NOW_SEND_SUCCESS;
    if (ok) s_ok++; else s_fail++;
    motion_on_ack(ok);
    // En unicast SUCCESS = ACK MAC del brazo; en broadcast solo indica que salió al aire.
    if (s_tx_verbose || !ok) ESP_LOGI(TAG, "tx %s en %lld us", ok ? "OK" : "FALLO", dt);
}

__attribute__((unused)) static esp_err_t add_peer(const uint8_t mac[6])
{
    if (esp_now_is_peer_exist(mac)) esp_now_del_peer(mac);
    esp_now_peer_info_t p = {.channel = 0, .ifidx = WIFI_IF_STA, .encrypt = false};
    memcpy(p.peer_addr, mac, 6);
    return esp_now_add_peer(&p);
}

__attribute__((unused)) static bool scan_arm(void)
{
    wifi_scan_config_t sc = {.ssid = (uint8_t *)CONFIG_ARM_AP_SSID, .show_hidden = false};
    if (esp_wifi_scan_start(&sc, true) != ESP_OK) return false;
    uint16_t n = 1;
    wifi_ap_record_t rec;
    if (esp_wifi_scan_get_ap_records(&n, &rec) != ESP_OK || n == 0) {
        esp_wifi_clear_ap_list();
        ESP_LOGW(TAG, "scan: \"%s\" no encontrado (¿brazo encendido?)", CONFIG_ARM_AP_SSID);
        return false;
    }
    esp_wifi_clear_ap_list();
    memcpy(s_arm_ap, rec.bssid, 6);
    s_channel = rec.primary;
    ESP_ERROR_CHECK(esp_wifi_set_channel(s_channel, WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(add_peer(s_arm_ap));
    s_arm_found = true;
    ESP_LOGI(TAG, "scan: \"%s\" " MACSTR " canal %u RSSI %d", CONFIG_ARM_AP_SSID, MAC2STR(s_arm_ap), s_channel, rec.rssi);
    return true;
}

// Rechaza lo que puede hacer daño con el firmware de fábrica.
static const char *unsafe(const char *j)
{
    if (strstr(j, "\"T\":0,") || strstr(j, "\"T\":0}")) return "T:0 quita el torque 10 s (el brazo cae)";
    if (strstr(j, "\"T\":1041")) return "T:1041 mueve a velocidad máxima";
    if (strstr(j, "\"spd\":0,") || strstr(j, "\"spd\":0}")) return "spd=0 es velocidad máxima";
    if (strstr(j, "\"acc\":0,") || strstr(j, "\"acc\":0}")) return "acc=0 es aceleración máxima";
    return NULL;
}

static bool send_raw(const char *json, bool verbose)
{
    const char *why = unsafe(json);
    if (why) { ESP_LOGE(TAG, "bloqueado: %s", why); return false; }
    if (strlen(json) >= sizeof(((roarm_espnow_msg_t *)0)->message)) { ESP_LOGE(TAG, "JSON demasiado largo"); return false; }
#if CONFIG_ARM_LINK_UART
    return arm_uart_send(json, !verbose, verbose);   // los pasos (no verbose) cuentan para motion
#endif
    if (s_unicast && !s_arm_found) { ESP_LOGE(TAG, "brazo no encontrado: usa 'scan' o 'peer bcast'"); return false; }

    roarm_espnow_msg_t m = {0};
    m.cmd = s_cmd;
    strcpy(m.message, json);
    const uint8_t *dst = s_unicast ? s_arm_ap : BCAST;
    s_tx_verbose = verbose;
    s_tx_us = esp_timer_get_time();
    esp_err_t e = esp_now_send(dst, (const uint8_t *)&m, sizeof m);
    s_sent++;
    if (verbose || e != ESP_OK)
        ESP_LOGI(TAG, "-> [%s cmd=%u] %s%s", s_unicast ? "ap" : "bcast", s_cmd, json, e == ESP_OK ? "" : "  (esp_now_send falló)");
    return e == ESP_OK;
}

// Los pasos de motion.c: sin log por paso (20 por segundo).
static bool send_step(const char *json)
{
    return send_raw(json, false);
}

static void log_status(void)
{
    motion_status_t m = motion_status();
#if CONFIG_ARM_LINK_UART
    arm_uart_status_t u = arm_uart_status();
    ESP_LOGI(TAG, "enlace UART | bytes rx %lu | ecos %lu sin eco %lu | eco último %.1f ms máx %.1f ms | otras líneas %lu",
             u.rx_bytes, u.echoes, u.timeouts, u.echo_ms_last, u.echo_ms_max, u.other_lines);
    if (u.valid)
        ESP_LOGI(TAG, "posición REAL del brazo (hace %lld ms): b=%+.3f s=%+.3f e=%+.3f t=%+.3f",
                 u.age_ms, u.pose.b, u.pose.s, u.pose.e, u.pose.h);
    else
        ESP_LOGW(TAG, "sin lectura de posición del brazo (¿cable TX/RX/GND?)");
#else
    ESP_LOGI(TAG, "brazo %s " MACSTR " canal %u, peer=%s cmd=%u | tx %lu ok %lu fallo %lu",
             s_arm_found ? "encontrado" : "NO encontrado", MAC2STR(s_arm_ap), s_channel,
             s_unicast ? "ap" : "bcast", s_cmd, s_sent, s_ok, s_fail);
#endif
    ESP_LOGI(TAG, "motion %s%s vmax %.0f°/s | pose b=%+.3f s=%+.3f e=%+.3f h=%+.3f | destino b=%+.3f s=%+.3f e=%+.3f h=%+.3f | pasos %lu",
             motion_state_name(m.state), m.frozen ? " (FROZEN)" : "", m.vmax * 57.2958f,
             m.cmd.b, m.cmd.s, m.cmd.e, m.cmd.h, m.target.b, m.target.s, m.target.e, m.target.h, m.steps);
}

static void handle_line(char *line)
{
    const char *why = NULL;
    if (!strncmp(line, "onset stop", 10)) {
        actions_stop();
    } else if (!strncmp(line, "onset", 5)) {
        int n = 20;
        float amp = 20;
        sscanf(line + 5, "%d %f", &n, &amp);
        actions_onset_test(n, amp);
#if CONFIG_ARM_LINK_UART
    } else if (!strncmp(line, "probe", 5)) {
        int pin = CONFIG_ARM_LINK_RX_GPIO;
        sscanf(line + 5, "%d", &pin);
        if (pin != CONFIG_ARM_LINK_RX_GPIO && pin != CONFIG_ARM_LINK_TX_GPIO) ESP_LOGE(TAG, "probe: solo GPIO%d o GPIO%d", CONFIG_ARM_LINK_RX_GPIO, CONFIG_ARM_LINK_TX_GPIO);
        else arm_uart_probe(pin);
#endif
    } else if (!strncmp(line, "synctest", 8)) {
        actions_sync_test();
    } else if (!strncmp(line, "led", 3)) {
        actions_led_test();
    } else if (!strncmp(line, "s3 ", 3)) {
        safety_sim_s3(strncmp(line + 3, "off", 3) ? atoi(line + 3) : -1);
    } else if (!strncmp(line, "confirm", 7)) {
        safety_confirm();
    } else if (!strncmp(line, "reset", 5)) {
        safety_reset();
    } else if (!strncmp(line, "home", 4)) {
        actions_disarm();
        if (!motion_home(&why)) ESP_LOGE(TAG, "home rechazado: %s", why);
    } else if (!strncmp(line, "gob ", 4)) {
        actions_disarm();
        motion_status_t m = motion_status();
        pose_t p = m.cmd;
        if (sscanf(line + 4, "%f", &p.b) != 1) { ESP_LOGE(TAG, "uso: gob <rad>"); return; }
        if (!motion_set_target(p, &why)) ESP_LOGE(TAG, "gob rechazado: %s", why);
        else ESP_LOGI(TAG, "destino base %+.3f", p.b);
    } else if (!strncmp(line, "act ", 4)) {
        pose_t p;
        int label = ARM_LABEL_UNLABELED;
        if (sscanf(line + 4, "%f %f %f %f %d", &p.b, &p.s, &p.e, &p.h, &label) < 4) { ESP_LOGE(TAG, "uso: act <b> <s> <e> <h> [etiqueta] (rad)"); return; }
        if (!actions_act(p, (uint8_t)label, &why)) ESP_LOGE(TAG, "act rechazado: %s", why);
        else ESP_LOGI(TAG, "acción: destino b=%+.3f s=%+.3f e=%+.3f h=%+.3f", p.b, p.s, p.e, p.h);
    } else if (!strncmp(line, "go ", 3)) {
        actions_disarm();
        pose_t p;
        if (sscanf(line + 3, "%f %f %f %f", &p.b, &p.s, &p.e, &p.h) != 4) { ESP_LOGE(TAG, "uso: go <b> <s> <e> <h> (rad)"); return; }
        if (!motion_set_target(p, &why)) ESP_LOGE(TAG, "go rechazado: %s", why);
        else ESP_LOGI(TAG, "destino b=%+.3f s=%+.3f e=%+.3f h=%+.3f", p.b, p.s, p.e, p.h);
    } else if (!strncmp(line, "vel ", 4)) {
        motion_set_vmax(strtof(line + 4, NULL) / 57.2958f);
        ESP_LOGI(TAG, "vmax %.1f °/s", motion_status().vmax * 57.2958f);
    } else if (!strncmp(line, "stop", 4)) {
        actions_disarm();
        motion_stop();
        ESP_LOGW(TAG, "STOP");
    } else if (!strncmp(line, "freeze", 6)) {
        motion_freeze(true);
        ESP_LOGW(TAG, "FREEZE: sin transmitir (simula supervisor muerto)");
    } else if (!strncmp(line, "resume", 6)) {
        motion_freeze(false);
        ESP_LOGI(TAG, "resume");
    } else if (!strncmp(line, "json ", 5)) {
        uint8_t st = safety_state();
        if (st != ARM_SAFETY_RUN && st != ARM_SAFETY_SLOW) ESP_LOGE(TAG, "json rechazado: seguridad en %s", safety_state_name(st));
        else if (send_raw(line + 5, true)) motion_invalidate();
#if !CONFIG_ARM_LINK_UART
    } else if (!strncmp(line, "scan", 4)) {
        scan_arm();
    } else if (!strncmp(line, "cmd ", 4)) {
        int c = atoi(line + 4);
        if (c == 1 || c == 2) s_cmd = c;
        ESP_LOGI(TAG, "cmd=%u", s_cmd);
    } else if (!strncmp(line, "peer ", 5)) {
        s_unicast = strncmp(line + 5, "bcast", 5) != 0;
        if (!s_unicast) add_peer(BCAST);
        ESP_LOGI(TAG, "peer=%s", s_unicast ? "ap" : "bcast");
#endif
    } else if (!strncmp(line, "det", 3)) {
#if CONFIG_ARM_LOCAL_DETECTOR
        detector_log_status();
#else
        ESP_LOGW(TAG, "detector local desactivado (CONFIG_ARM_LOCAL_DETECTOR)");
#endif
    } else if (!strncmp(line, "status", 6)) {
        log_status();
        safety_log_status();
#if CONFIG_ARM_LOCAL_DETECTOR
        detector_log_status();
#endif
    } else if (line[0]) {
        ESP_LOGW(TAG, "comando desconocido: %s", line);
    }
}

// Busca el brazo en segundo plano hasta encontrarlo (la consola y la seguridad ya funcionan).
__attribute__((unused)) static void scan_task(void *arg)
{
    while (!scan_arm()) vTaskDelay(pdMS_TO_TICKS(2000));
    vTaskDelete(NULL);
}

static void console_task(void *arg)
{
    char line[256];
    size_t n = 0;
    for (;;) {
        uint8_t c;
        if (uart_read_bytes(UART_NUM_0, &c, 1, portMAX_DELAY) != 1) continue;
        if (c == '\r' || c == '\n') {
            line[n] = 0;
            n = 0;
            handle_line(line);
        } else if (n < sizeof line - 1) {
            line[n++] = c;
        }
    }
}

void app_main(void)
{
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
#if !CONFIG_ARM_LINK_UART
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));
#endif

    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 512, 0, 0, NULL, 0));
    ESP_LOGI(TAG, "INTUNE supervisor, etapa 3 (niveles + heartbeat). Comandos: act, onset, led, s3, confirm, reset, home, go, gob, vel, stop, freeze, resume, json, scan, cmd, peer, status");

    motion_init(send_step);
    safety_init();
#if CONFIG_ARM_LOCAL_DETECTOR
    detector_init();
#endif
    actions_init();
    xTaskCreate(console_task, "console", 4096, NULL, 5, NULL);
#if CONFIG_ARM_LINK_UART
    arm_uart_init();
#else
    xTaskCreate(scan_task, "scan", 4096, NULL, 4, NULL);
#endif
}
