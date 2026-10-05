#include "safety.h"

#include <math.h>
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "motion.h"

static const char *TAG = "safety";

#define S3_UART        UART_NUM_2
#define TICK_MS        20
#define LONG_PRESS_MS  2000
#define DEBOUNCE_MS    50

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static uint8_t s_state = ARM_SAFETY_NO_S3;
static bool s_ever_alive;
static int64_t s_last_alert_us;
static uint8_t s_level_rx = 0xFF;
static uint8_t s_reason_rx;
static bool s_have_seq;
static uint16_t s_last_seq;
static uint32_t s_alerts_rx, s_alerts_lost;
static bool s_confirm_req, s_reset_req;

static arm_rx_t s_rx;        // UART real
static arm_rx_t s_rx_sim;    // simulación desde consola
static volatile int s_sim_level = -1;
static volatile bool s_led_override;

const char *safety_state_name(uint8_t st)
{
    switch (st) {
    case ARM_SAFETY_NO_S3:   return "NO_S3";
    case ARM_SAFETY_RUN:     return "RUN";
    case ARM_SAFETY_SLOW:    return "SLOW";
    case ARM_SAFETY_PAUSED:  return "PAUSED";
    case ARM_SAFETY_STOPPED: return "STOPPED";
    }
    return "?";
}

static void on_frame(uint8_t type, const uint8_t *payload, uint8_t len, void *ctx)
{
    arm_alert_t a;
    if (type != ARM_TYPE_ALERT || !arm_decode_alert(payload, len, &a)) return;   // otros tipos: se ignoran
    uint8_t prev;
    portENTER_CRITICAL(&s_mux);
    if (s_have_seq) {
        uint16_t d = (uint16_t)(a.seq - s_last_seq);
        if (d > 1 && d < 0x8000) s_alerts_lost += d - 1u;
    }
    s_have_seq = true;
    s_last_seq = a.seq;
    s_alerts_rx++;
    s_last_alert_us = esp_timer_get_time();
    s_ever_alive = true;
    prev = s_level_rx;
    s_level_rx = a.level;
    s_reason_rx = a.reason;
    portEXIT_CRITICAL(&s_mux);
    if (prev != a.level) ESP_LOGI(TAG, "S3: nivel %u (razón %u, score %.3f)", a.level, a.reason, a.score);
}

static void uart_rx_task(void *arg)
{
    uint8_t buf[64];
    for (;;) {
        int n = uart_read_bytes(S3_UART, buf, sizeof buf, pdMS_TO_TICKS(20));
        if (n > 0) arm_rx_feed(&s_rx, buf, (size_t)n, on_frame, NULL);
    }
}

static void sim_task(void *arg)
{
    uint16_t seq = 0;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(ARM_HEARTBEAT_MS));
        int lvl = s_sim_level;
        if (lvl < 0) continue;
        arm_alert_t a = {.level = (uint8_t)lvl, .reason = ARM_REASON_MANUAL, .seq = seq++,
                         .t_ms = (uint32_t)(esp_timer_get_time() / 1000), .score = NAN};
        uint8_t f[ARM_MAX_FRAME];
        size_t n = arm_encode_alert(f, &a);
        arm_rx_feed(&s_rx_sim, f, n, on_frame, NULL);
    }
}

static void apply(uint8_t st)
{
    switch (st) {
    case ARM_SAFETY_RUN:
        motion_set_speed_scale(1.0f);
        motion_hold(false, false);
        break;
    case ARM_SAFETY_SLOW:
        motion_set_speed_scale(CONFIG_ARM_LEVEL1_SPEED_PCT / 100.0f);
        motion_hold(false, false);
        break;
    case ARM_SAFETY_PAUSED:
        motion_hold(true, true);    // retoma el destino al confirmar
        break;
    default:                        // STOPPED, NO_S3
        motion_hold(true, false);   // el destino se descarta
        break;
    }
}

static void send_status(void)
{
    motion_status_t m = motion_status();
    arm_status_t st;
    portENTER_CRITICAL(&s_mux);
    st = (arm_status_t){
        .safety = s_state, .level_rx = s_level_rx, .motion = (uint8_t)m.state,
        .flags = (uint8_t)((m.state == MOTION_LINK_LOST ? ARM_STF_ARM_LINK_LOST : 0) |
                           (m.state == MOTION_UNKNOWN ? ARM_STF_POSE_UNKNOWN : 0)),
        .alerts_rx = s_alerts_rx, .alerts_lost = s_alerts_lost,
        .crc_errors = s_rx.crc_errors, .espnow_fail = m.fails,
    };
    portEXIT_CRITICAL(&s_mux);
    uint8_t f[ARM_MAX_FRAME];
    size_t n = arm_encode_status(f, &st);
    uart_write_bytes(S3_UART, f, n);
}

static void set_led(uint8_t st, uint32_t tick)
{
    if (s_led_override) return;
    uint32_t ms = tick * TICK_MS;
    int on;
    switch (st) {
    case ARM_SAFETY_RUN:     on = 1; break;                        // fijo
    case ARM_SAFETY_SLOW:    on = (ms % 1000) < 500; break;        // 1 Hz
    case ARM_SAFETY_PAUSED:  on = (ms % 400) < 200; break;         // 2.5 Hz
    case ARM_SAFETY_STOPPED: on = (ms % 200) < 100; break;         // 5 Hz
    default:                 on = (ms % 2000) < 100; break;        // destello cada 2 s: sin S3
    }
    gpio_set_level(CONFIG_ARM_LED_GPIO, on);
}

static void safety_task(void *arg)
{
    uint32_t tick = 0;
    int64_t press_start = -1;
    bool long_fired = false;
    int64_t last_status_us = 0;
    TickType_t last = xTaskGetTickCount();

    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(TICK_MS));
        tick++;
        int64_t now = esp_timer_get_time();

        // Botón BOOT (activo en bajo): corta = confirmar, larga = rearmar.
        bool pressed = gpio_get_level(CONFIG_ARM_BUTTON_GPIO) == 0;
        if (pressed && press_start < 0) {
            press_start = now;
            long_fired = false;
        } else if (pressed && !long_fired && now - press_start >= LONG_PRESS_MS * 1000LL) {
            long_fired = true;
            safety_reset();
        } else if (!pressed && press_start >= 0) {
            if (!long_fired && now - press_start >= DEBOUNCE_MS * 1000LL) safety_confirm();
            press_start = -1;
        }

        uint8_t prev, st;
        bool confirm, reset, alive;
        uint8_t lvl;
        portENTER_CRITICAL(&s_mux);
        alive = s_ever_alive && now - s_last_alert_us < ARM_HEARTBEAT_TIMEOUT_MS * 1000LL;
        lvl = s_level_rx;
        confirm = s_confirm_req;
        reset = s_reset_req;
        s_confirm_req = s_reset_req = false;
        prev = st = s_state;
        uint8_t run_st = lvl == ARM_LEVEL_NORMAL ? ARM_SAFETY_RUN : ARM_SAFETY_SLOW;

        if (!alive) {
            if (st != ARM_SAFETY_NO_S3) st = ARM_SAFETY_STOPPED;   // perder el heartbeat enclava la parada
        } else if (lvl == ARM_LEVEL_SEVERE) {
            st = ARM_SAFETY_STOPPED;
        } else if (st == ARM_SAFETY_STOPPED) {
            if (reset && lvl <= ARM_LEVEL_MILD) st = run_st;
        } else if (lvl == ARM_LEVEL_MODERATE) {
            st = ARM_SAFETY_PAUSED;
        } else if (st == ARM_SAFETY_PAUSED) {
            if (confirm) st = run_st;                               // aquí lvl <= 1
        } else {
            st = run_st;                                            // NO_S3 al arrancar, RUN, SLOW
        }
        s_state = st;
        portEXIT_CRITICAL(&s_mux);

        if (st != prev) {
            apply(st);
            ESP_LOGW(TAG, "%s -> %s (nivel S3 %d, heartbeat %s)", safety_state_name(prev), safety_state_name(st),
                     lvl == 0xFF ? -1 : lvl, alive ? "vivo" : "PERDIDO");
        }
        if (confirm && st == prev)
            ESP_LOGW(TAG, "confirmar ignorado (estado %s, nivel %d)", safety_state_name(st), lvl == 0xFF ? -1 : lvl);
        if (reset && st == prev)
            ESP_LOGW(TAG, "rearme ignorado (estado %s, nivel %d, heartbeat %s): requiere nivel <= 1 y S3 vivo",
                     safety_state_name(st), lvl == 0xFF ? -1 : lvl, alive ? "vivo" : "perdido");

        set_led(st, tick);
        if (st != prev || now - last_status_us >= ARM_STATUS_PERIOD_MS * 1000LL) {
            send_status();
            last_status_us = now;
        }
    }
}

void safety_init(void)
{
    arm_rx_init(&s_rx);
    arm_rx_init(&s_rx_sim);

    uart_config_t uc = {
        .baud_rate = ARM_BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(S3_UART, 512, 512, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(S3_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(S3_UART, CONFIG_ARM_S3_UART_TX_GPIO, CONFIG_ARM_S3_UART_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    gpio_config_t led = {.pin_bit_mask = 1ULL << CONFIG_ARM_LED_GPIO, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&led));
    gpio_config_t btn = {.pin_bit_mask = 1ULL << CONFIG_ARM_BUTTON_GPIO, .mode = GPIO_MODE_INPUT,
                         .pull_up_en = GPIO_PULLUP_ENABLE};
    ESP_ERROR_CHECK(gpio_config(&btn));

    apply(ARM_SAFETY_NO_S3);   // nada se mueve hasta el primer heartbeat
    xTaskCreate(uart_rx_task, "s3_rx", 3072, NULL, 12, NULL);
    xTaskCreate(sim_task, "s3_sim", 3072, NULL, 11, NULL);
    xTaskCreate(safety_task, "safety", 4096, NULL, 13, NULL);
    ESP_LOGI(TAG, "UART S3 %d baud TX=%d RX=%d, heartbeat timeout %d ms. Sin S3: brazo retenido.",
             ARM_BAUD, CONFIG_ARM_S3_UART_TX_GPIO, CONFIG_ARM_S3_UART_RX_GPIO, ARM_HEARTBEAT_TIMEOUT_MS);
}

void safety_sim_s3(int level)
{
    s_sim_level = level > ARM_LEVEL_SEVERE ? ARM_LEVEL_SEVERE : level;
    if (level < 0) ESP_LOGW(TAG, "simulación S3 detenida (S3 \"muerto\")");
    else ESP_LOGI(TAG, "simulación S3: nivel %d cada %d ms", s_sim_level, ARM_HEARTBEAT_MS);
}

void safety_led_override(bool on)
{
    s_led_override = on;
    if (on) gpio_set_level(CONFIG_ARM_LED_GPIO, 0);
}

void safety_send_event(const arm_event_t *e)
{
    uint8_t f[ARM_MAX_FRAME];
    size_t n = arm_encode_event(f, e);
    uart_write_bytes(S3_UART, f, n);
}

void safety_confirm(void)
{
    portENTER_CRITICAL(&s_mux);
    s_confirm_req = true;
    portEXIT_CRITICAL(&s_mux);
}

void safety_reset(void)
{
    portENTER_CRITICAL(&s_mux);
    s_reset_req = true;
    portEXIT_CRITICAL(&s_mux);
}

uint8_t safety_state(void)
{
    return s_state;
}

void safety_log_status(void)
{
    portENTER_CRITICAL(&s_mux);
    uint8_t st = s_state, lvl = s_level_rx;
    uint32_t rx = s_alerts_rx, lost = s_alerts_lost, crc = s_rx.crc_errors;
    int64_t age = s_ever_alive ? (esp_timer_get_time() - s_last_alert_us) / 1000 : -1;
    portEXIT_CRITICAL(&s_mux);
    ESP_LOGI(TAG, "seguridad %s | nivel S3 %d | último ALERT hace %lld ms | ALERT %lu perdidos %lu CRC %lu | sim %d",
             safety_state_name(st), lvl == 0xFF ? -1 : lvl, age, rx, lost, crc, s_sim_level);
}
