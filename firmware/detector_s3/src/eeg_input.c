// Backend provisional: tramas Unicorn crudas (lo que reenvía hoy el puente).
// Aplica aquí lo que en la versión final hace C1: µV, huecos (regla dura 3) e
// IIR causal 1-15 Hz (regla dura 1). Cuando exista link_protocol.h, ese backend
// solo copia muestras y flags al ring.
#include "eeg_input.h"

#include <math.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "unicorn_parser.h"

static const char *TAG = "eeg_in";

#define LINK_UART       UART_NUM_1
#define LINK_MAX_HOLD   25    // muestras (100 ms): más que esto => reset del filtro
#define SETTLING_LEN    500   // 2 s tras cada reset
#define T0_BUCKET_US    (5 * 1000 * 1000)
#define SAMPLE_US       4000

static errp_ring_t *s_ring;
static SemaphoreHandle_t s_ring_lock;
static unicorn_parser_t s_parser;
static errp_iir_t s_iir[ERRP_N_CH];
static float s_last_uv[ERRP_N_CH];
static bool s_have_last;
static uint32_t s_last_cnt;
static int s_settling;         // muestras restantes marcadas SETTLING
static uint32_t s_resets;
static int64_t s_rx_time_us;   // llegada del bloque UART en curso
static volatile int64_t s_last_frame_us;
static float s_battery;

// t0 = mínimo de (llegada - contador x 4 ms) en el bloque actual y el anterior (5-10 s)
static portMUX_TYPE s_t0_mux = portMUX_INITIALIZER_UNLOCKED;
static int64_t s_t0_cur, s_t0_prev, s_t0_bucket_start;
static bool s_t0_cur_valid, s_t0_prev_valid;

static void t0_reset(void)
{
    portENTER_CRITICAL(&s_t0_mux);
    s_t0_cur_valid = s_t0_prev_valid = false;
    portEXIT_CRITICAL(&s_t0_mux);
}

static void t0_update(int64_t t_arrival, uint32_t counter)
{
    int64_t v = t_arrival - (int64_t)counter * SAMPLE_US;
    portENTER_CRITICAL(&s_t0_mux);
    if (!s_t0_cur_valid) {
        s_t0_cur = v;
        s_t0_cur_valid = true;
        s_t0_bucket_start = t_arrival;
    } else {
        if (t_arrival - s_t0_bucket_start > T0_BUCKET_US) {
            s_t0_prev = s_t0_cur;
            s_t0_prev_valid = true;
            s_t0_cur = v;
            s_t0_bucket_start = t_arrival;
        }
        if (v < s_t0_cur) {
            s_t0_cur = v;
        }
    }
    portEXIT_CRITICAL(&s_t0_mux);
}

bool eeg_input_counter_at(int64_t t_us, uint32_t *counter)
{
    portENTER_CRITICAL(&s_t0_mux);
    bool ok = s_t0_cur_valid;
    int64_t t0 = s_t0_cur;
    if (s_t0_prev_valid && s_t0_prev < t0) {
        t0 = s_t0_prev;
    }
    portEXIT_CRITICAL(&s_t0_mux);
    if (!ok || t_us < t0) {
        return false;
    }
    *counter = (uint32_t)((t_us - t0 + SAMPLE_US / 2) / SAMPLE_US);
    return true;
}

static void filter_reset(const float uv[ERRP_N_CH])
{
    for (int c = 0; c < ERRP_N_CH; c++) {
        errp_iir_reset(&s_iir[c], uv[c]);
    }
    s_settling = SETTLING_LEN;
    s_resets++;
}

static void push_sample(uint32_t cnt, const float uv[ERRP_N_CH], float gyr, uint8_t flags)
{
    float out[ERRP_N_CH];
    for (int c = 0; c < ERRP_N_CH; c++) {
        out[c] = errp_iir_step(&s_iir[c], uv[c]);
    }
    if (s_settling > 0) {
        s_settling--;
        flags |= ERRP_SF_SETTLING;
    }
    xSemaphoreTake(s_ring_lock, portMAX_DELAY);
    errp_ring_put(s_ring, cnt, out, gyr, flags);
    xSemaphoreGive(s_ring_lock);
}

static void on_frame(const uint8_t *frame, uint32_t gap, void *ctx)
{
    (void)ctx;
    unicorn_sample_t s;
    unicorn_decode(frame, &s);
    float gyr = 0;
    for (int i = 0; i < 3; i++) {
        gyr = fmaxf(gyr, fabsf(s.gyr_dps[i]));
    }
    s_battery = s.battery_pct;
    s_last_frame_us = s_rx_time_us;

    bool restart = !s_have_last || (int32_t)(s.counter - s_last_cnt) <= 0;  // sesión nueva
    if (restart || gap > LINK_MAX_HOLD) {
        if (restart) {
            t0_reset();
        }
        filter_reset(s.eeg_uv);
    } else if (gap > 0) {
        // Regla dura 3: retención de orden cero, nunca interpolación
        uint8_t f = gap == 1 ? ERRP_SF_HELD : ERRP_SF_GAP;
        for (uint32_t k = 1; k <= gap; k++) {
            push_sample(s_last_cnt + k, s_last_uv, 0, f);
        }
    }
    push_sample(s.counter, s.eeg_uv, gyr, 0);
    memcpy(s_last_uv, s.eeg_uv, sizeof(s_last_uv));
    s_last_cnt = s.counter;
    s_have_last = true;
    t0_update(s_rx_time_us, s.counter);
}

static void rx_task(void *arg)
{
    (void)arg;
    uint8_t buf[512];
    for (;;) {
        int n = uart_read_bytes(LINK_UART, buf, sizeof(buf), pdMS_TO_TICKS(20));
        if (n > 0) {
            s_rx_time_us = esp_timer_get_time();
            unicorn_parser_feed(&s_parser, buf, (size_t)n, on_frame, NULL);
        }
    }
}

bool eeg_input_ok(int64_t now_us)
{
#if CONFIG_DETECTOR_BRIDGE_STATUS_GPIO >= 0
    if (!gpio_get_level(CONFIG_DETECTOR_BRIDGE_STATUS_GPIO)) {
        return false;
    }
#endif
    int64_t last = s_last_frame_us;
    return last != 0 && now_us - last < (int64_t)CONFIG_DETECTOR_EEG_TIMEOUT_MS * 1000;
}

void eeg_input_get_stats(eeg_input_stats_t *o)
{
    o->frames = s_parser.frames;
    o->gaps = s_parser.gaps;
    o->lost = s_parser.lost;
    o->corrupt = s_parser.corrupt;
    o->discarded = s_parser.discarded;
    o->filter_resets = s_resets;
    o->battery_pct = s_battery;
    uint32_t dummy;
    o->t0_valid = eeg_input_counter_at(esp_timer_get_time(), &dummy);
}

void eeg_input_start(errp_ring_t *ring, SemaphoreHandle_t ring_lock)
{
    s_ring = ring;
    s_ring_lock = ring_lock;
    unicorn_parser_init(&s_parser);

#if CONFIG_DETECTOR_BRIDGE_STATUS_GPIO >= 0
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << CONFIG_DETECTOR_BRIDGE_STATUS_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,  // puente desconectado => no streaming
    };
    ESP_ERROR_CHECK(gpio_config(&io));
#endif
    uart_config_t uc = {
        .baud_rate = CONFIG_DETECTOR_LINK_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(LINK_UART, 8192, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(LINK_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(LINK_UART, UART_PIN_NO_CHANGE, CONFIG_DETECTOR_LINK_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    xTaskCreatePinnedToCore(rx_task, "eeg_rx", 4096, NULL, 12, NULL, 0);
    ESP_LOGI(TAG, "UART%d RX GPIO%d @ %d, tramas Unicorn crudas + IIR 1-15 Hz en el S3 (provisional)",
             LINK_UART, CONFIG_DETECTOR_LINK_RX_GPIO, CONFIG_DETECTOR_LINK_BAUD);
}
