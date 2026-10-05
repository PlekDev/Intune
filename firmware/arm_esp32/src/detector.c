#include "detector.h"

#include <math.h>
#include <string.h>
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "arm_protocol.h"
#include "autoencoder_engine.h"
#include "link_protocol.h"
#include "safety.h"

static const char *TAG = "det";

#define DET_UART        UART_NUM_2          // la que usaba el enlace con el S3 (safety.c no la instala)
#define RING_N          1024                // ~4 s a 250 Hz
#define PRE             AE_PRE_SAMPLES      // 50: [-200, 0) ms
#define WIN             AE_WIN_SAMPLES      // 250: [-200, +800) ms
#define WAIT_AFTER      250                 // procesar cuando llegue t0 + 1 s: así se ve un OVERLAP posterior
#define MAX_PEND        8
#define EEG_TIMEOUT_US  1000000             // sin tramas EEG en 1 s = EEG perdido (el BT tiene huecos de cientos de ms)
// Puerta de artefactos: mismos valores y definiciones que ml/data/build_dataset.py (C4)
#define GATE_UV         100.0f              // max |época| tras línea base y diezmado
#define GATE_GYRO_DPS   30.0f               // rango pico a pico por eje de gyr en la ventana
#define FLAT_STD_UV     0.05f               // desviación estándar mínima por canal de la época
#define OVERLAP_SAMPLES 250                 // otro flanco a < 1.0 s
#define REJECT_FLAGS    (LINK_F_HELD | LINK_F_GAP | LINK_F_SETTLING)
#define GYR_SCALE       (1.0f / 32.8f)      // int16 Unicorn -> °/s

typedef struct {
    uint32_t cnt;
    uint8_t flags;
    float x[LINK_N_CH];
    int16_t gyr[3];
} samp_t;

typedef struct {
    uint32_t cnt;
    bool overlap;
} pend_t;

enum { R_FLAGS, R_COUNTER, R_GYRO, R_AMPLITUDE, R_FLAT, R_OVERLAP, R_NO_T0, R_INVALID, R_N };
static const char *R_NAMES[R_N] = {"flags", "counter", "gyro", "amplitude", "flat", "overlap", "no_t0", "invalid"};

// Todo esto lo toca solo det_task (salvo los contadores que lee detector_log_status)
static samp_t s_ring[RING_N];
static uint32_t s_wr;                  // muestras escritas en total
static pend_t s_pend[MAX_PEND];
static int s_npend;
static bool s_have_evt;
static uint32_t s_last_evt_cnt;
static link_rx_t s_rx;
static float s_win[AE_N_CH][WIN];      // ventana de la época (8 KB, fuera de la pila)
static float s_epoch[AE_N_CH][AE_N_T];
static ae_workspace_t s_ws;
static ae_calib_t s_cal;
static int s_rej_streak, s_t2_streak;

static volatile int64_t s_last_eeg_us;
static volatile uint8_t s_bridge_state = 0xFF;
static volatile bool s_ever_alive;
static volatile uint8_t s_level, s_reason = ARM_REASON_NONE;
static volatile float s_last_score = NAN;
static volatile uint32_t s_n_epochs, s_n_scored, s_n_rej[R_N], s_last_infer_us, s_max_infer_us, s_events;
static volatile uint32_t s_lvl_count[4];
static volatile uint32_t s_max_gap_us, s_eeg_lost_n;
static bool s_was_alive;

static const samp_t *ring_find(uint32_t cnt)
{
    uint32_t n = s_wr < RING_N ? s_wr : RING_N;
    for (uint32_t k = 1; k <= n; k++) {
        const samp_t *s = &s_ring[(s_wr - k) % RING_N];
        if (s->cnt == cnt) return s;
        if (s->cnt < cnt) return NULL;  // más viejo que lo buscado
    }
    return NULL;
}

static void reject(int why, uint32_t cnt)
{
    s_n_rej[why]++;
    s_rej_streak++;
    if (s_rej_streak >= 3 && s_level < ARM_LEVEL_MODERATE) {
        // CLAUDE.md: 3 épocas rechazadas seguidas = operador no observable -> nivel 2
        s_level = ARM_LEVEL_MODERATE;
        s_reason = ARM_REASON_NONE;
    }
    ESP_LOGI(TAG, "época cnt=%lu RECHAZADA (%s), racha %d -> nivel %u", (unsigned long)cnt, R_NAMES[why],
             s_rej_streak, s_level);
}

static void process_epoch(const pend_t *p)
{
    s_n_epochs++;
    uint32_t c0 = p->cnt - PRE;
    const samp_t *first = ring_find(c0);
    if (!first) { reject(R_COUNTER, p->cnt); return; }
    size_t i0 = (size_t)(first - s_ring);
    int16_t gmin[3] = {INT16_MAX, INT16_MAX, INT16_MAX}, gmax[3] = {INT16_MIN, INT16_MIN, INT16_MIN};
    bool bad_flags = false;
    for (int j = 0; j < WIN; j++) {
        const samp_t *s = &s_ring[(i0 + j) % RING_N];
        if (s->cnt != c0 + (uint32_t)j) { reject(R_COUNTER, p->cnt); return; }
        if (s->flags & REJECT_FLAGS) bad_flags = true;
        for (int ch = 0; ch < AE_N_CH; ch++) s_win[ch][j] = s->x[ch];
        for (int a = 0; a < 3; a++) {
            if (s->gyr[a] < gmin[a]) gmin[a] = s->gyr[a];
            if (s->gyr[a] > gmax[a]) gmax[a] = s->gyr[a];
        }
    }
    if (p->overlap) { reject(R_OVERLAP, p->cnt); return; }
    if (bad_flags) { reject(R_FLAGS, p->cnt); return; }
    float gyro = 0;
    for (int a = 0; a < 3; a++) {
        float r = (gmax[a] - gmin[a]) * GYR_SCALE;
        if (r > gyro) gyro = r;
    }
    if (gyro > GATE_GYRO_DPS) { reject(R_GYRO, p->cnt); return; }

    ae_preprocess((const float (*)[WIN])s_win, s_epoch);
    float amax = 0;
    for (int ch = 0; ch < AE_N_CH; ch++) {
        float m = 0, v = 0;
        for (int k = 0; k < AE_N_T; k++) {
            float a = fabsf(s_epoch[ch][k]);
            if (a > amax) amax = a;
            m += s_epoch[ch][k];
        }
        m /= AE_N_T;
        for (int k = 0; k < AE_N_T; k++) v += (s_epoch[ch][k] - m) * (s_epoch[ch][k] - m);
        if (sqrtf(v / AE_N_T) < FLAT_STD_UV) { reject(R_FLAT, p->cnt); return; }
    }
    if (amax > GATE_UV) { reject(R_AMPLITUDE, p->cnt); return; }

    int64_t t0 = esp_timer_get_time();
    ae_result_t r = ae_infer((const float (*)[AE_N_T])s_epoch, &s_cal, &s_ws);
    uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
    s_last_infer_us = us;
    if (us > s_max_infer_us) s_max_infer_us = us;
    if (!r.valid) { reject(R_INVALID, p->cnt); s_level = ARM_LEVEL_SEVERE; s_reason = ARM_REASON_ERRP; return; }

    // Lógica de alertas de CLAUDE.md (por acción). safety.c enclava la pausa (2) y la parada (3).
    s_rej_streak = 0;
    s_n_scored++;
    s_last_score = r.score;
    s_t2_streak = r.score > s_cal.t2 ? s_t2_streak + 1 : 0;
    uint8_t lvl;
    if (r.score > s_cal.t3 || s_t2_streak >= 2) lvl = ARM_LEVEL_SEVERE;
    else if (r.score > s_cal.t2) lvl = ARM_LEVEL_MODERATE;
    else if (r.score > s_cal.t1) lvl = ARM_LEVEL_MILD;
    else lvl = ARM_LEVEL_NORMAL;
    s_level = lvl;
    s_reason = lvl ? ARM_REASON_ERRP : ARM_REASON_NONE;
    s_lvl_count[lvl]++;
    ESP_LOGI(TAG, "época cnt=%lu score=%.3f (T1 %.3f T2 %.3f T3 %.3f) -> nivel %u  [%lu us]", (unsigned long)p->cnt,
             r.score, s_cal.t1, s_cal.t2, s_cal.t3, lvl, (unsigned long)us);
}

static void on_frame(uint8_t type, const uint8_t *payload, uint8_t len, void *ctx)
{
    if (type == LINK_TYPE_EEG) {
        link_eeg_t e;
        if (!link_decode_eeg(payload, len, &e)) return;
        if (e.flags & LINK_F_SESSION_START) {  // el contador vuelve a empezar: lo pendiente ya no vale
            for (int i = 0; i < s_npend; i++) reject(R_COUNTER, s_pend[i].cnt);
            s_npend = 0;
            s_have_evt = false;
        }
        samp_t *s = &s_ring[s_wr % RING_N];
        s->cnt = e.counter;
        s->flags = e.flags;
        memcpy(s->x, e.eeg_uv, sizeof s->x);
        memcpy(s->gyr, e.gyr, sizeof s->gyr);
        s_wr++;
        int64_t now = esp_timer_get_time();
        if (s_last_eeg_us && now - s_last_eeg_us > (int64_t)s_max_gap_us) s_max_gap_us = (uint32_t)(now - s_last_eeg_us);
        s_last_eeg_us = now;
    } else if (type == LINK_TYPE_STATUS) {
        link_status_t st;
        if (!link_decode_status(payload, len, &st)) return;
        s_bridge_state = st.state;
        if (st.state == LINK_STATE_STREAMING && s_wr > 0 && !s_ever_alive) {
            s_ever_alive = true;
            ESP_LOGW(TAG, "EEG del puente recibido: el detector empieza a mandar niveles");
        }
    } else if (type == LINK_TYPE_EVENT && len == sizeof(link_event_t)) {
        link_event_t ev;
        memcpy(&ev, payload, sizeof ev);
        s_events++;
        if (ev.flags & LINK_EVT_F_NO_T0) { s_n_epochs++; reject(R_NO_T0, 0); return; }
        bool overlap = (ev.flags & LINK_EVT_F_OVERLAP) ||
                       (s_have_evt && ev.counter - s_last_evt_cnt < OVERLAP_SAMPLES);
        if (overlap && s_npend && s_pend[s_npend - 1].cnt == s_last_evt_cnt) s_pend[s_npend - 1].overlap = true;
        s_have_evt = true;
        s_last_evt_cnt = ev.counter;
        if (s_npend == MAX_PEND) { s_n_epochs++; reject(R_COUNTER, s_pend[0].cnt); memmove(s_pend, s_pend + 1, (MAX_PEND - 1) * sizeof(pend_t)); s_npend--; }
        s_pend[s_npend++] = (pend_t){.cnt = ev.counter, .overlap = overlap};
    }
}

static void det_task(void *arg)
{
    static uint8_t buf[512];
    uint16_t seq = 0;
    int64_t next_hb = esp_timer_get_time();
    for (;;) {
        int n = uart_read_bytes(DET_UART, buf, sizeof buf, pdMS_TO_TICKS(10));
        if (n > 0) link_rx_feed(&s_rx, buf, (size_t)n, on_frame, NULL);

        // Épocas cuya ventana (y 1 s para ver un OVERLAP posterior) ya llegó
        uint32_t newest = s_wr ? s_ring[(s_wr - 1) % RING_N].cnt : 0;
        while (s_npend && newest >= s_pend[0].cnt + WAIT_AFTER) {
            pend_t p = s_pend[0];
            memmove(s_pend, s_pend + 1, (size_t)(s_npend - 1) * sizeof(pend_t));
            s_npend--;
            process_epoch(&p);
        }

        // Heartbeat hacia safety.c (cada ARM_HEARTBEAT_MS), como el ALERT del S3
        int64_t now = esp_timer_get_time();
        if (now >= next_hb) {
            next_hb = now + ARM_HEARTBEAT_MS * 1000LL;
            if (!s_ever_alive || safety_sim_active()) continue;  // antes del primer EEG: brazo retenido (NO_S3)
            bool alive = now - s_last_eeg_us < EEG_TIMEOUT_US && s_bridge_state == LINK_STATE_STREAMING;
            if (s_was_alive && !alive) {
                s_eeg_lost_n++;
                ESP_LOGW(TAG, "EEG PERDIDO (última trama hace %lld ms, puente %s) -> nivel 3",
                         (now - s_last_eeg_us) / 1000, s_bridge_state == LINK_STATE_STREAMING ? "STREAMING" : "sin streaming");
            }
            s_was_alive = alive;
            arm_alert_t a = {.level = alive ? s_level : ARM_LEVEL_SEVERE,
                             .reason = alive ? s_reason : ARM_REASON_EEG_LOST, .seq = seq++,
                             .t_ms = (uint32_t)(now / 1000), .score = s_last_score};
            safety_inject_alert(&a);
        }
    }
}

void detector_init(void)
{
    link_rx_init(&s_rx);
    uart_config_t uc = {
        .baud_rate = LINK_BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT,
    };
    // RX grande: una inferencia de unos ms no debe perder bytes (921600 baud ~ 27 kB/s con EEG_RAW)
    ESP_ERROR_CHECK(uart_driver_install(DET_UART, 16384, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(DET_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(DET_UART, UART_PIN_NO_CHANGE, CONFIG_ARM_S3_UART_RX_GPIO, UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE));

    bool ok = ae_init();
    ae_calib_default(&s_cal);  // normalización y umbrales del entrenamiento (sin calibración por sesión)
    const ae_model_info_t *mi = ae_model_info();
    if (ok) {
        ESP_LOGI(TAG, "ErrP-AE %s (%s%s) listo: arena %lu/%lu B, T1 %.3f T2 %.3f T3 %.3f", mi->model_id, mi->arch,
                 mi->placeholder ? ", PLACEHOLDER" : "", (unsigned long)mi->arena_used_bytes,
                 (unsigned long)mi->arena_bytes, s_cal.t1, s_cal.t2, s_cal.t3);
    } else {
        ESP_LOGE(TAG, "ae_init falló: cada época dará nivel 3 (fail-safe)");
    }
    ESP_LOGI(TAG, "EEG del puente por UART RX=GPIO%d a %d baud; sin EEG el brazo queda retenido",
             CONFIG_ARM_S3_UART_RX_GPIO, LINK_BAUD);
    // core 1, por debajo de safety (13) y del movimiento
    xTaskCreatePinnedToCore(det_task, "det", 8192, NULL, 8, NULL, 1);
}

void detector_operator_ack(void)
{
    // Sin esto, tras un nivel 3 con el brazo parado no hay acciones ni épocas nuevas que bajen el
    // nivel, y safety.c nunca acepta el rearme (exige nivel <= 1): bloqueo.
    s_level = ARM_LEVEL_NORMAL;
    s_reason = ARM_REASON_NONE;
    s_rej_streak = 0;
    s_t2_streak = 0;
    int64_t now = esp_timer_get_time();
    bool alive = s_ever_alive && now - s_last_eeg_us < EEG_TIMEOUT_US && s_bridge_state == LINK_STATE_STREAMING;
    if (!alive || safety_sim_active()) return;
    // Entregar ya el nivel 0, antes de que safety_task evalúe la confirmación o el rearme
    arm_alert_t a = {.level = ARM_LEVEL_NORMAL, .reason = ARM_REASON_NONE, .seq = 0,
                     .t_ms = (uint32_t)(now / 1000), .score = s_last_score};
    safety_inject_alert(&a);
    ESP_LOGI(TAG, "operador confirmó / rearmó: nivel del detector a 0");
}

void detector_log_status(void)
{
    int64_t age = s_last_eeg_us ? (esp_timer_get_time() - s_last_eeg_us) / 1000 : -1;
    ESP_LOGI(TAG, "detector | EEG hace %lld ms, puente %s | eventos %lu épocas %lu puntuadas %lu | nivel %u | "
             "último score %.3f | inferencia %lu us (máx %lu)", age,
             s_bridge_state == LINK_STATE_STREAMING ? "STREAMING" : "sin streaming", (unsigned long)s_events,
             (unsigned long)s_n_epochs, (unsigned long)s_n_scored, s_level, s_last_score,
             (unsigned long)s_last_infer_us, (unsigned long)s_max_infer_us);
    ESP_LOGI(TAG, "  enlace: tramas %lu CRC mal %lu bytes descartados %lu | hueco máx entre muestras %lu ms | "
             "EEG perdido %lu veces", (unsigned long)s_rx.frames, (unsigned long)s_rx.crc_errors,
             (unsigned long)s_rx.discarded, (unsigned long)(s_max_gap_us / 1000), (unsigned long)s_eeg_lost_n);
    ESP_LOGI(TAG, "  niveles 0/1/2/3: %lu/%lu/%lu/%lu | rechazos flags %lu counter %lu gyro %lu amplitude %lu "
             "flat %lu overlap %lu no_t0 %lu invalid %lu",
             (unsigned long)s_lvl_count[0], (unsigned long)s_lvl_count[1], (unsigned long)s_lvl_count[2],
             (unsigned long)s_lvl_count[3], (unsigned long)s_n_rej[R_FLAGS], (unsigned long)s_n_rej[R_COUNTER],
             (unsigned long)s_n_rej[R_GYRO], (unsigned long)s_n_rej[R_AMPLITUDE], (unsigned long)s_n_rej[R_FLAT],
             (unsigned long)s_n_rej[R_OVERLAP], (unsigned long)s_n_rej[R_NO_T0], (unsigned long)s_n_rej[R_INVALID]);
}
