// INTUNE C3: supervisor del brazo (ESP32 aparte).
//
// Etapa 2: consola por USB + movimiento por pasos cortos (motion.c) hacia el RoArm-M2-S por ESP-NOW.
// El brazo conserva su firmware de fábrica; no se toca su ESP32.
//
// Comandos (línea + Enter, 115200):
//   home                      movimiento lento a la postura inicial; después se sabe dónde está el brazo
//   go <b> <s> <e> <h>        destino en rad (base, hombro, codo, pinza), enviado por pasos
//   gob <b>                   solo la base
//   vel <grados/s>            velocidad máxima de los pasos (por defecto 20)
//   stop                      frena: destino = posición actual
//   freeze / resume           prueba: deja de transmitir de golpe (supervisor "muerto") / reanuda
//   json <JSON>               JSON crudo (bloquea T:0, T:1041 y spd/acc 0); invalida la posición
//   scan                      busca el AP del brazo, fija canal y peer
//   cmd <1|2>                 campo cmd del mensaje (2 = loop() del brazo, por defecto)
//   peer <ap|bcast>           unicast a la MAC del AP del brazo (con ACK, por defecto) o broadcast
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
#include "motion.h"
#include "roarm_espnow.h"

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

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    int64_t dt = esp_timer_get_time() - s_tx_us;
    bool ok = status == ESP_NOW_SEND_SUCCESS;
    if (ok) s_ok++; else s_fail++;
    motion_on_ack(ok);
    // En unicast SUCCESS = ACK MAC del brazo; en broadcast solo indica que salió al aire.
    if (s_tx_verbose || !ok) ESP_LOGI(TAG, "tx %s en %lld us", ok ? "OK" : "FALLO", dt);
}

static esp_err_t add_peer(const uint8_t mac[6])
{
    if (esp_now_is_peer_exist(mac)) esp_now_del_peer(mac);
    esp_now_peer_info_t p = {.channel = 0, .ifidx = WIFI_IF_STA, .encrypt = false};
    memcpy(p.peer_addr, mac, 6);
    return esp_now_add_peer(&p);
}

static bool scan_arm(void)
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
    ESP_LOGI(TAG, "brazo %s " MACSTR " canal %u, peer=%s cmd=%u | tx %lu ok %lu fallo %lu",
             s_arm_found ? "encontrado" : "NO encontrado", MAC2STR(s_arm_ap), s_channel,
             s_unicast ? "ap" : "bcast", s_cmd, s_sent, s_ok, s_fail);
    ESP_LOGI(TAG, "motion %s%s vmax %.0f°/s | pose b=%+.3f s=%+.3f e=%+.3f h=%+.3f | destino b=%+.3f s=%+.3f e=%+.3f h=%+.3f | pasos %lu",
             motion_state_name(m.state), m.frozen ? " (FROZEN)" : "", m.vmax * 57.2958f,
             m.cmd.b, m.cmd.s, m.cmd.e, m.cmd.h, m.target.b, m.target.s, m.target.e, m.target.h, m.steps);
}

static void handle_line(char *line)
{
    const char *why = NULL;
    if (!strncmp(line, "home", 4)) {
        motion_home();
    } else if (!strncmp(line, "gob ", 4)) {
        motion_status_t m = motion_status();
        pose_t p = m.cmd;
        if (sscanf(line + 4, "%f", &p.b) != 1) { ESP_LOGE(TAG, "uso: gob <rad>"); return; }
        if (!motion_set_target(p, &why)) ESP_LOGE(TAG, "gob rechazado: %s", why);
        else ESP_LOGI(TAG, "destino base %+.3f", p.b);
    } else if (!strncmp(line, "go ", 3)) {
        pose_t p;
        if (sscanf(line + 3, "%f %f %f %f", &p.b, &p.s, &p.e, &p.h) != 4) { ESP_LOGE(TAG, "uso: go <b> <s> <e> <h> (rad)"); return; }
        if (!motion_set_target(p, &why)) ESP_LOGE(TAG, "go rechazado: %s", why);
        else ESP_LOGI(TAG, "destino b=%+.3f s=%+.3f e=%+.3f h=%+.3f", p.b, p.s, p.e, p.h);
    } else if (!strncmp(line, "vel ", 4)) {
        motion_set_vmax(strtof(line + 4, NULL) / 57.2958f);
        ESP_LOGI(TAG, "vmax %.1f °/s", motion_status().vmax * 57.2958f);
    } else if (!strncmp(line, "stop", 4)) {
        motion_stop();
        ESP_LOGW(TAG, "STOP");
    } else if (!strncmp(line, "freeze", 6)) {
        motion_freeze(true);
        ESP_LOGW(TAG, "FREEZE: sin transmitir (simula supervisor muerto)");
    } else if (!strncmp(line, "resume", 6)) {
        motion_freeze(false);
        ESP_LOGI(TAG, "resume");
    } else if (!strncmp(line, "json ", 5)) {
        if (send_raw(line + 5, true)) motion_invalidate();
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
    } else if (!strncmp(line, "status", 6)) {
        log_status();
    } else if (line[0]) {
        ESP_LOGW(TAG, "comando desconocido: %s", line);
    }
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
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));

    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 512, 0, 0, NULL, 0));
    ESP_LOGI(TAG, "INTUNE supervisor, etapa 2 (pasos cortos). Comandos: home, go, gob, vel, stop, freeze, resume, json, scan, cmd, peer, status");

    for (int i = 0; i < 3 && !scan_arm(); i++) vTaskDelay(pdMS_TO_TICKS(1000));
    motion_init(send_step);
    xTaskCreate(console_task, "console", 4096, NULL, 5, NULL);
}
