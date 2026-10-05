// Entrada de EEG desde el puente (C1) por UART.
//   DETECTOR_INPUT_LINK (default): tramas de firmware/common/link_protocol.h; el puente
//     ya filtró, rellenó huecos y marcó flags (reglas duras 1 y 3): solo se copian al ring.
//     STATUS (1/s) es el heartbeat del puente.
//   DETECTOR_INPUT_RAW_UNICORN (respaldo): tramas Unicorn crudas; el S3 hace aquí lo
//     mismo que el puente (µV, huecos, IIR 1-15 Hz) con los mismos flags.
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
#if CONFIG_DETECTOR_INPUT_LINK
#include "link_protocol.h"
#endif

static const char *TAG = "eeg_in";

#define LINK_UART       UART_NUM_1
#define SAMPLE_US       4000
#define T0_BUCKET_US    (5 * 1000 * 1000)
#define STATUS_TIMEOUT_US (2500 * 1000)  // STATUS llega 1/s

static errp_ring_t *s_ring;
static SemaphoreHandle_t s_ring_lock;
static int64_t s_rx_time_us;            // llegada del bloque UART en curso
static volatile int64_t s_last_frame_us;
static float s_battery;
static uint32_t s_frames, s_gaps, s_lost, s_resets, s_corrupt, s_discarded;
static bool s_have_last;
static uint32_t s_last_cnt;

// ---------- t0 = mínimo de (llegada - contador x 4 ms) en el bloque actual y el anterior ----------
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

static void ring_put(uint32_t cnt, const float eeg[ERRP_N_CH], const float gyr[3], uint8_t flags)
{
    xSemaphoreTake(s_ring_lock, portMAX_DELAY);
    errp_ring_put(s_ring, cnt, eeg, gyr, flags);
    xSemaphoreGive(s_ring_lock);
}

// Contador nuevo: sesión nueva (vuelve atrás) => t0 se reestima; huecos solo como estadística
static void track_counter(uint32_t cnt, bool session_start)
{
    if (session_start || (s_have_last && (int32_t)(cnt - s_last_cnt) <= 0)) {
        t0_reset();
    } else if (s_have_last && cnt - s_last_cnt > 1) {
        s_gaps++;
        s_lost += cnt - s_last_cnt - 1;
    }
    s_last_cnt = cnt;
    s_have_last = true;
    s_frames++;
    s_last_frame_us = s_rx_time_us;
    t0_update(s_rx_time_us, cnt);
}

#if CONFIG_DETECTOR_INPUT_LINK
// ================= link_protocol.h =================
static link_rx_t s_link;
static volatile int64_t s_last_status_us;
static volatile uint8_t s_bridge_state = LINK_STATE_IDLE;

static void on_link_frame(uint8_t type, const uint8_t *payload, uint8_t len, void *ctx)
{
    (void)ctx;
    if (type == LINK_TYPE_EEG) {
        link_eeg_t s;
        if (!link_decode_eeg(payload, len, &s)) {
            return;
        }
        float gyr[3], eeg[ERRP_N_CH];
        for (int a = 0; a < 3; a++) {
            gyr[a] = s.gyr[a] * UNICORN_GYR_SCALE_DPS;
        }
        memcpy(eeg, (const void *)s.eeg_uv, sizeof(eeg));  // miembro packed: sin punteros desalineados
        // HELD/GAP/SETTLING/FILTER_RESET/UNFILTERED usan los mismos bits que ERRP_SF_*
        ring_put(s.counter, eeg, gyr, s.flags);
        if (!(s.flags & LINK_F_HELD)) {  // las rellenadas no llegaron en ese instante
            track_counter(s.counter, s.flags & LINK_F_SESSION_START);
        }
        if (s.flags & LINK_F_FILTER_RESET) {
            s_resets++;
        }
    } else if (type == LINK_TYPE_STATUS) {
        link_status_t st;
        if (link_decode_status(payload, len, &st)) {
            s_bridge_state = st.state;
            s_battery = st.battery_pct == 0xFF ? NAN : st.battery_pct;
            s_last_status_us = esp_timer_get_time();
        }
    }
    // EEG_RAW, EVENT y tipos desconocidos: se ignoran
}

static void feed(const uint8_t *buf, int n)
{
    link_rx_feed(&s_link, buf, (size_t)n, on_link_frame, NULL);
    s_corrupt = s_link.crc_errors;
    s_discarded = s_link.discarded;
}

static bool bridge_ok(int64_t now)
{
    return s_bridge_state == LINK_STATE_STREAMING && now - s_last_status_us < STATUS_TIMEOUT_US;
}

static void input_init(void)
{
    link_rx_init(&s_link);
}

#else
// ================= tramas Unicorn crudas (provisional) =================
#define LINK_MAX_HOLD 25   // muestras (100 ms): más que esto => reset del filtro
#define SETTLING_LEN  500  // 2 s tras cada reset

static unicorn_parser_t s_parser;
static errp_iir_t s_iir[ERRP_N_CH];
static float s_last_uv[ERRP_N_CH];
static int s_settling;

static void push_filtered(uint32_t cnt, const float uv[ERRP_N_CH], const float gyr[3], uint8_t flags)
{
    float out[ERRP_N_CH];
    for (int c = 0; c < ERRP_N_CH; c++) {
        out[c] = errp_iir_step(&s_iir[c], uv[c]);
    }
    if (s_settling > 0) {
        s_settling--;
        flags |= ERRP_SF_SETTLING;
    }
    ring_put(cnt, out, gyr, flags);
}

static void on_unicorn_frame(const uint8_t *frame, uint32_t gap, void *ctx)
{
    (void)ctx;
    unicorn_sample_t s;
    unicorn_decode(frame, &s);
    s_battery = s.battery_pct;
    bool restart = !s_have_last || (int32_t)(s.counter - s_last_cnt) <= 0;
    uint8_t flags = 0;
    if (restart || gap > LINK_MAX_HOLD) {
        for (int c = 0; c < ERRP_N_CH; c++) {
            errp_iir_reset(&s_iir[c], s.eeg_uv[c]);
        }
        s_settling = SETTLING_LEN;
        s_resets++;
        flags = ERRP_SF_FILTER_RESET;
    } else if (gap > 0) {
        // Regla dura 3: retención de orden cero, nunca interpolación
        uint8_t f = gap == 1 ? ERRP_SF_HELD : (ERRP_SF_HELD | ERRP_SF_GAP);
        float g0[3] = {0};
        for (uint32_t k = 1; k <= gap; k++) {
            push_filtered(s_last_cnt + k, s_last_uv, g0, f);
        }
    }
    push_filtered(s.counter, s.eeg_uv, s.gyr_dps, flags);
    memcpy(s_last_uv, s.eeg_uv, sizeof(s_last_uv));
    track_counter(s.counter, restart);
}

static void feed(const uint8_t *buf, int n)
{
    unicorn_parser_feed(&s_parser, buf, (size_t)n, on_unicorn_frame, NULL);
    s_corrupt = s_parser.corrupt;
    s_discarded = s_parser.discarded;
}

static bool bridge_ok(int64_t now)
{
    (void)now;
    return true;  // solo GPIO de estado y llegada de tramas
}

static void input_init(void)
{
    unicorn_parser_init(&s_parser);
}
#endif

static void rx_task(void *arg)
{
    (void)arg;
    uint8_t buf[512];
    for (;;) {
        int n = uart_read_bytes(LINK_UART, buf, sizeof(buf), pdMS_TO_TICKS(20));
        if (n > 0) {
            s_rx_time_us = esp_timer_get_time();
            feed(buf, n);
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
    return bridge_ok(now_us) && last != 0 && now_us - last < (int64_t)CONFIG_DETECTOR_EEG_TIMEOUT_MS * 1000;
}

void eeg_input_get_stats(eeg_input_stats_t *o)
{
    o->frames = s_frames;
    o->gaps = s_gaps;
    o->lost = s_lost;
    o->corrupt = s_corrupt;
    o->discarded = s_discarded;
    o->filter_resets = s_resets;
    o->battery_pct = s_battery;
    uint32_t dummy;
    o->t0_valid = eeg_input_counter_at(esp_timer_get_time(), &dummy);
}

void eeg_input_start(errp_ring_t *ring, SemaphoreHandle_t ring_lock)
{
    s_ring = ring;
    s_ring_lock = ring_lock;
    input_init();

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
#if CONFIG_DETECTOR_INPUT_LINK
    ESP_LOGI(TAG, "UART%d RX GPIO%d @ %d, link_protocol.h (EEG filtrado por el puente)",
             LINK_UART, CONFIG_DETECTOR_LINK_RX_GPIO, CONFIG_DETECTOR_LINK_BAUD);
#else
    ESP_LOGW(TAG, "UART%d RX GPIO%d @ %d, tramas Unicorn crudas + IIR 1-15 Hz en el S3 (provisional)",
             LINK_UART, CONFIG_DETECTOR_LINK_RX_GPIO, CONFIG_DETECTOR_LINK_BAUD);
#endif
}
