#include "actions.h"

#include <math.h>
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "motion.h"
#include "safety.h"

static const char *TAG = "actions";

static esp_timer_handle_t s_sync_off, s_led_off;
static volatile bool s_armed;           // la próxima salida IDLE -> MOVING es una acción
static uint16_t s_action_id;
static uint8_t s_label = ARM_LABEL_UNLABELED;
static uint32_t s_event_seq;
static volatile bool s_running, s_abort;
static int s_reps;
static float s_amp;

static void sync_off(void *arg) { gpio_set_level(CONFIG_ARM_SYNC_GPIO, 0); }
static void led_off(void *arg) { gpio_set_level(CONFIG_ARM_LED_GPIO, 0); }

// Desde la tarea de movimiento, justo antes del primer paso. No bloquea.
static void on_onset(void)
{
    if (!s_armed) return;
    s_armed = false;
    gpio_set_level(CONFIG_ARM_SYNC_GPIO, 1);
    gpio_set_level(CONFIG_ARM_LED_GPIO, 1);
    int64_t t = esp_timer_get_time();
    esp_timer_start_once(s_sync_off, ARM_SYNC_PULSE_US);
    esp_timer_start_once(s_led_off, ACTIONS_LED_MS * 1000);
    uint8_t st = safety_state();
    arm_event_t e = {.seq = s_event_seq++, .action_id = s_action_id, .kind = ARM_EVENT_ONSET,
                     .label = s_label, .t_us = (uint32_t)t,
                     .level = st == ARM_SAFETY_SLOW ? ARM_LEVEL_MILD : ARM_LEVEL_NORMAL};
    safety_send_event(&e);
}

static bool wait_idle(int timeout_ms)
{
    for (int t = 0; t < timeout_ms; t += 20) {
        if (s_abort) return false;
        motion_status_t m = motion_status();
        if (m.state == MOTION_IDLE && !m.held) return true;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    return false;
}

static void onset_task(void *arg)
{
    motion_status_t m = motion_status();
    pose_t home = m.cmd, away = m.cmd;
    away.b += (home.b <= 0 ? 1 : -1) * s_amp / 57.2958f;
    safety_led_override(true);
    ESP_LOGW(TAG, "prueba de inicio: %d acciones de %.0f° en la base, intervalo 2-4 s aleatorio", s_reps, s_amp);

    int done = 0;
    for (int i = 0; i < s_reps && !s_abort; i++) {
        // Intervalo aleatorio: el operador (y la cámara) no pueden anticipar el inicio.
        int gap = 2000 + (int)(esp_random() % 2001);
        vTaskDelay(pdMS_TO_TICKS(gap));
        uint8_t st = safety_state();
        if (st != ARM_SAFETY_RUN && st != ARM_SAFETY_SLOW) {
            ESP_LOGE(TAG, "abortada: seguridad en %s", safety_state_name(st));
            break;
        }
        const char *why;
        s_action_id = (uint16_t)i;
        s_label = ARM_LABEL_UNLABELED;
        s_armed = true;
        if (!motion_set_target((i % 2 == 0) ? away : home, &why)) {
            s_armed = false;
            ESP_LOGE(TAG, "abortada: %s", why);
            break;
        }
        if (!wait_idle(10000)) {
            ESP_LOGE(TAG, "abortada: el movimiento no terminó");
            break;
        }
        ESP_LOGI(TAG, "acción %d/%d hecha (espera previa %d ms)", i + 1, s_reps, gap);
        done++;
    }
    // Volver al punto de partida si quedó en el otro extremo.
    if (motion_status().cmd.b != home.b) {
        const char *why;
        vTaskDelay(pdMS_TO_TICKS(1000));
        motion_set_target(home, &why);
        wait_idle(10000);
    }
    s_armed = false;
    safety_led_override(false);
    ESP_LOGW(TAG, "prueba de inicio terminada: %d/%d acciones", done, s_reps);
    s_running = false;
    vTaskDelete(NULL);
}

void actions_onset_test(int reps, float amp_deg)
{
    if (s_running) { ESP_LOGE(TAG, "ya hay una prueba en marcha"); return; }
    uint8_t st = safety_state();
    motion_status_t m = motion_status();
    if (st != ARM_SAFETY_RUN && st != ARM_SAFETY_SLOW) { ESP_LOGE(TAG, "requiere seguridad RUN/SLOW (está en %s)", safety_state_name(st)); return; }
    if (m.state != MOTION_IDLE) { ESP_LOGE(TAG, "requiere movimiento IDLE (está en %s; ¿falta home?)", motion_state_name(m.state)); return; }
    if (reps < 1) reps = 20;
    if (amp_deg < 5 || amp_deg > 45) amp_deg = 20;
    s_reps = reps;
    s_amp = amp_deg;
    s_abort = false;
    s_running = true;
    xTaskCreate(onset_task, "onset", 4096, NULL, 6, NULL);
}

void actions_stop(void)
{
    s_abort = true;
    s_armed = false;
    motion_stop();
}

void actions_led_test(void)
{
    safety_led_override(true);
    for (int i = 0; i < 3; i++) {
        gpio_set_level(CONFIG_ARM_LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(300));
        gpio_set_level(CONFIG_ARM_LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(300));
    }
    safety_led_override(false);
}

void actions_init(void)
{
    gpio_config_t g = {.pin_bit_mask = 1ULL << CONFIG_ARM_SYNC_GPIO, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&g));
    gpio_set_level(CONFIG_ARM_SYNC_GPIO, 0);
    const esp_timer_create_args_t a = {.callback = sync_off, .name = "sync_off"};
    const esp_timer_create_args_t b = {.callback = led_off, .name = "led_off"};
    ESP_ERROR_CHECK(esp_timer_create(&a, &s_sync_off));
    ESP_ERROR_CHECK(esp_timer_create(&b, &s_led_off));
    motion_set_onset_cb(on_onset);
}
