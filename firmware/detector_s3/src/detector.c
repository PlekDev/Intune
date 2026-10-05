#include "detector.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "arm_link.h"
#include "driver/gpio.h"
#include "eeg_input.h"
#include "errp_dsp.h"
#include "errp_golden.h"
#include "errp_model.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs.h"
#include "sdkconfig.h"

static const char *TAG = "detector";

#define N_CALIB      CONFIG_DETECTOR_CALIB_EPOCHS
#define OVERLAP_US   (ARM_SYNC_MIN_SPACING_MS * 1000LL)
#define MAX_PENDING  4
#define NVS_NS       "intune"
#define NVS_KEY      "calib"
#define CALIB_MAGIC  0x494E5431u  // "INT1"

typedef struct {
    uint32_t magic;
    float mean[ERRP_N_CH], std[ERRP_N_CH];
    float t_ae[3], t_lda[3];
    uint16_t n;
} calib_t;

typedef struct {
    int64_t t_us;       // flanco de sincronía
    uint32_t counter;   // muestra t = 0
    bool counter_ok;
    bool overlap;
} pulse_t;

static errp_ring_t s_ring;
static SemaphoreHandle_t s_ring_lock;
static QueueHandle_t s_sync_q;
static calib_t s_cal;
static errp_alert_t s_alert;

// Calibración en curso
static bool s_calibrating;
static int s_calib_n;
static float (*s_calib_epochs)[ERRP_N_CH][ERRP_N_T];
static float s_calib_lda[150];

// ---------- sincronía ----------
static void IRAM_ATTR sync_isr(void *arg)
{
    (void)arg;
    int64_t t = esp_timer_get_time();  // la ISR solo marca el tiempo
    BaseType_t woken = pdFALSE;
    xQueueSendFromISR(s_sync_q, &t, &woken);
    if (woken) {
        portYIELD_FROM_ISR();
    }
}

// ---------- calibración ----------
static void calib_defaults(void)
{
    s_cal.magic = CALIB_MAGIC;
    memcpy(s_cal.mean, ERRP_DEFAULT_MEAN, sizeof(s_cal.mean));
    memcpy(s_cal.std, ERRP_DEFAULT_STD, sizeof(s_cal.std));
    s_cal.t_ae[0] = ERRP_DEFAULT_T1;
    s_cal.t_ae[1] = ERRP_DEFAULT_T2;
    s_cal.t_ae[2] = ERRP_DEFAULT_T3;
    // Sin umbrales LDA offline de correctos de sesión: usar los del entrenamiento LDA no
    // aplica en vivo; se fuerzan a calibración. Hasta entonces, el AE decide.
    s_cal.t_lda[0] = s_cal.t_lda[1] = s_cal.t_lda[2] = INFINITY;
    s_cal.n = 0;
}

static bool calib_load(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK) {
        return false;
    }
    calib_t c;
    size_t len = sizeof(c);
    bool ok = nvs_get_blob(h, NVS_KEY, &c, &len) == ESP_OK && len == sizeof(c) && c.magic == CALIB_MAGIC;
    nvs_close(h);
    if (ok) {
        s_cal = c;
    }
    return ok;
}

static void calib_save(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_blob(h, NVS_KEY, &s_cal, sizeof(s_cal));
        nvs_commit(h);
        nvs_close(h);
    }
}

static const float *active_thresholds(void)
{
#if CONFIG_DETECTOR_USE_LDA
    return s_cal.t_lda;
#else
    return s_cal.t_ae;
#endif
}

static void alert_reset_thresholds(void)
{
    const float *t = active_thresholds();
    int level = s_alert.level;
    errp_alert_init(&s_alert, t[0], t[1], t[2]);
    s_alert.level = level;
}

static void calib_begin(void)
{
    if (!s_calib_epochs) {
        size_t sz = sizeof(*s_calib_epochs) * N_CALIB;
        s_calib_epochs = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
        if (!s_calib_epochs) {
            s_calib_epochs = heap_caps_malloc(sz, MALLOC_CAP_8BIT);
        }
        if (!s_calib_epochs) {
            ESP_LOGE(TAG, "sin memoria para calibrar (%u B): se quedan los umbrales actuales", (unsigned)sz);
            return;
        }
    }
    s_calib_n = 0;
    s_calibrating = true;
    ESP_LOGW(TAG, "CALIBRANDO: el operador observa %d acciones correctas", N_CALIB);
    printf("{\"ev\":\"calib_start\",\"n\":%d}\n", N_CALIB);
}

static void calib_finish(void)
{
    // mean/std por canal sobre epochs y tiempo (errp_pipeline.channel_stats)
    for (int c = 0; c < ERRP_N_CH; c++) {
        double s = 0, s2 = 0;
        for (int i = 0; i < s_calib_n; i++) {
            for (int t = 0; t < ERRP_N_T; t++) {
                double v = s_calib_epochs[i][c][t];
                s += v;
                s2 += v * v;
            }
        }
        double n = (double)s_calib_n * ERRP_N_T;
        double m = s / n;
        double sd = sqrt(fmax(s2 / n - m * m, 0.0));
        s_cal.mean[c] = (float)m;
        s_cal.std[c] = fmaxf((float)sd, ERRP_STD_FLOOR);  // piso como build_dataset.py
    }
    static float scores[150];
    for (int i = 0; i < s_calib_n; i++) {
        errp_normalize(s_calib_epochs[i], s_cal.mean, s_cal.std);
        scores[i] = errp_model_score(s_calib_epochs[i], NULL);
    }
    const float pct[3] = {ERRP_PCT_T1, ERRP_PCT_T2, ERRP_PCT_T3};
    for (int k = 0; k < 3; k++) {
        s_cal.t_ae[k] = errp_percentile(scores, s_calib_n, pct[k]);
        s_cal.t_lda[k] = errp_percentile(s_calib_lda, s_calib_n, pct[k]);
    }
    s_cal.n = (uint16_t)s_calib_n;
    calib_save();
    s_calibrating = false;
    alert_reset_thresholds();
    s_alert.level = 0;  // se calibró con EEG limpio: arrancar en normal
    ESP_LOGW(TAG, "calibración lista (%d epochs): AE T1 %.4f T2 %.4f T3 %.4f | LDA T1 %.3f T2 %.3f T3 %.3f",
             s_calib_n, s_cal.t_ae[0], s_cal.t_ae[1], s_cal.t_ae[2], s_cal.t_lda[0], s_cal.t_lda[1], s_cal.t_lda[2]);
    printf("{\"ev\":\"calib_done\",\"n\":%d,\"t_ae\":[%.5f,%.5f,%.5f],\"t_lda\":[%.4f,%.4f,%.4f]}\n", s_calib_n,
           s_cal.t_ae[0], s_cal.t_ae[1], s_cal.t_ae[2], s_cal.t_lda[0], s_cal.t_lda[1], s_cal.t_lda[2]);
}

// ---------- epoch ----------
static void process_pulse(const pulse_t *p, uint16_t action_id, uint8_t *flags_out, float *score_out)
{
    static float win[ERRP_N_CH][ERRP_EPOCH_LEN];
    static float e[ERRP_N_CH][ERRP_N_T];
    errp_epoch_status_t st = ERRP_EPOCH_COUNTER;
    errp_window_info_t info = {0};
    if (p->counter_ok) {
        xSemaphoreTake(s_ring_lock, portMAX_DELAY);
        st = errp_epoch_cut(&s_ring, p->counter, win, &info);
        xSemaphoreGive(s_ring_lock);
    }
    if (st == ERRP_EPOCH_OK) {
        errp_preprocess(win, e);  // X [8][40] µV, como el dataset de C4
        st = errp_epoch_gate(&info, e, (float)CONFIG_DETECTOR_GATE_GYRO_DPS);
    }
    uint8_t flags = s_calibrating ? ARM_FLAG_CALIBRATING : 0;
    float ae = NAN, lda = NAN;
    const char *status = p->overlap ? "overlap" : errp_epoch_status_str(st);

    if (p->overlap || st != ERRP_EPOCH_OK) {
        flags |= ARM_FLAG_EPOCH_REJECTED | (p->overlap ? ARM_FLAG_OVERLAP : 0);
        if (!s_calibrating) {
            errp_alert_rejected(&s_alert);
        }
    } else {
        lda = errp_lda_score(e);
        if (s_calibrating) {
            memcpy(s_calib_epochs[s_calib_n], e, sizeof(e));
            s_calib_lda[s_calib_n] = lda;
            if (++s_calib_n >= N_CALIB) {
                calib_finish();
            }
        } else {
            errp_normalize(e, s_cal.mean, s_cal.std);
            ae = errp_model_score(e, NULL);
#if CONFIG_DETECTOR_USE_LDA
            errp_alert_score(&s_alert, lda);
#else
            errp_alert_score(&s_alert, ae);
#endif
        }
    }
    printf("{\"ev\":\"epoch\",\"t_us\":%" PRId64 ",\"cnt\":%" PRIu32 ",\"action\":%u,\"status\":\"%s\","
           "\"ae\":%.5f,\"lda\":%.4f,\"level\":%d,\"flags\":%u,\"calib\":%d,\"inf_us\":%" PRId64 "}\n",
           p->t_us, p->counter, action_id, status, ae, lda, s_alert.level, flags,
           s_calibrating ? s_calib_n : -1, errp_model_last_us());
    *flags_out = flags;
    *score_out = ae;
}

static bool button_pressed(void)
{
#if CONFIG_DETECTOR_CALIB_BUTTON_GPIO >= 0
    return gpio_get_level(CONFIG_DETECTOR_CALIB_BUTTON_GPIO) == 0;
#else
    return false;
#endif
}

static void detector_task(void *arg)
{
    (void)arg;
    pulse_t pending[MAX_PENDING];
    int n_pending = 0;
    int64_t last_pulse_us = INT64_MIN / 2;
    uint16_t action_id = 0xFFFF;
    uint8_t flags = 0;
    float score = NAN;
    int64_t level2_since = 0, last_stats = 0;
    bool was_pressed = false;

    for (;;) {
        int64_t now = esp_timer_get_time();

        // Pulsos nuevos -> contador Unicorn de t = 0
        int64_t t_edge;
        while (xQueueReceive(s_sync_q, &t_edge, 0) == pdTRUE) {
            pulse_t p = {.t_us = t_edge};
            p.counter_ok = eeg_input_counter_at(t_edge - CONFIG_DETECTOR_EVENT_LATENCY_OFFSET_US, &p.counter);
            if (t_edge - last_pulse_us < OVERLAP_US) {
                p.overlap = true;
                for (int i = 0; i < n_pending; i++) {
                    if (t_edge - pending[i].t_us < OVERLAP_US) {
                        pending[i].overlap = true;  // también el anterior aún sin procesar
                    }
                }
            }
            last_pulse_us = t_edge;
            if (n_pending == MAX_PENDING) {
                memmove(pending, pending + 1, sizeof(pending[0]) * (MAX_PENDING - 1));
                n_pending--;
                ESP_LOGW(TAG, "demasiados pulsos pendientes: se descarta el más viejo");
            }
            pending[n_pending++] = p;
        }

        arm_event_t ev;
        while (arm_link_pop_event(&ev)) {
            action_id = ev.action_id;  // deliberate_error: LOG ONLY, solo telemetría
            printf("{\"ev\":\"arm_event\",\"action\":%u,\"type\":%u,\"deliberate_error\":%u}\n",
                   ev.action_id, ev.action_type, ev.deliberate_error);
        }

        // Procesar el pulso más viejo cuando ya llegaron sus +800 ms (o si nunca llegarán)
        if (n_pending > 0) {
            pulse_t *p = &pending[0];
            bool ready = !p->counter_ok;
            if (p->counter_ok) {
                xSemaphoreTake(s_ring_lock, portMAX_DELAY);
                ready = s_ring.any && (int32_t)(s_ring.newest - (p->counter + ERRP_POST - 1)) >= 0;
                xSemaphoreGive(s_ring_lock);
                ready |= now - p->t_us > 3 * 1000 * 1000;  // el EEG se cortó: rechazar
            }
            if (ready) {
                process_pulse(p, action_id, &flags, &score);
                memmove(pending, pending + 1, sizeof(pending[0]) * (n_pending - 1));
                n_pending--;
            }
        }

        // Confirmación del operador y timeout de la pausa
        if (arm_link_take_confirm()) {
            errp_alert_confirm(&s_alert);
            printf("{\"ev\":\"confirm\"}\n");
        }
        if (s_alert.level == 2) {
            if (level2_since == 0) {
                level2_since = now;
            } else if (now - level2_since > ARM_CONFIRM_TIMEOUT_MS * 1000LL) {
                s_alert.level = 3;
                ESP_LOGW(TAG, "pausa sin CONFIRM en %d ms: paro seguro", ARM_CONFIRM_TIMEOUT_MS);
            }
        } else {
            level2_since = 0;
        }

        // Recalibrar con el botón
        bool pressed = button_pressed();
        if (pressed && !was_pressed && !s_calibrating) {
            calib_begin();
        }
        was_pressed = pressed;

        // Nivel de salida. Fail-safe: sin EEG válido => 3, y al volver el EEG sigue en 3
        // hasta un epoch limpio <= T1 o CONFIRM. Calibrando: 0 (el operador debe ver
        // acciones correctas) salvo fail-safe.
        uint8_t out_flags = flags & ~ARM_FLAG_CALIBRATING;
        int out_level = s_alert.level;
        if (s_calibrating) {
            out_flags |= ARM_FLAG_CALIBRATING;
            out_level = 0;
        }
        if (!eeg_input_ok(now)) {
            s_alert.level = 3;
            out_level = 3;
            out_flags |= ARM_FLAG_EEG_LOST;
        }
        arm_link_set_alert((uint8_t)out_level, out_flags, action_id, score);

        if (now - last_stats > 5 * 1000 * 1000) {
            last_stats = now;
            eeg_input_stats_t es;
            arm_link_stats_t as;
            eeg_input_get_stats(&es);
            arm_link_get_stats(&as);
            printf("{\"ev\":\"stats\",\"frames\":%" PRIu32 ",\"gaps\":%" PRIu32 ",\"lost\":%" PRIu32
                   ",\"corrupt\":%" PRIu32 ",\"resets\":%" PRIu32 ",\"batt\":%.0f,\"t0\":%d,\"eeg_ok\":%d"
                   ",\"level\":%d,\"arm_tx\":%" PRIu32 ",\"arm_rx\":%" PRIu32 ",\"arm_crc\":%" PRIu32
                   ",\"heap\":%u}\n",
                   es.frames, es.gaps, es.lost, es.corrupt, es.filter_resets, es.battery_pct, es.t0_valid,
                   eeg_input_ok(now), s_alert.level, as.tx_frames, as.rx_frames, as.rx_crc_errors,
                   (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

bool detector_selftest(void)
{
    bool ok = true;
    for (int i = 0; i < ERRP_N_GOLDEN; i++) {
        // Misma ruta que en vivo: X µV -> LDA y normalización por canal -> modelo
        static float z[ERRP_N_CH][ERRP_N_T];
        memcpy(z, ERRP_GOLDEN_X[i], sizeof(z));
        float lda = errp_lda_score(z);
        errp_normalize(z, ERRP_GOLDEN_MEAN[i], ERRP_GOLDEN_STD[i]);
        float s = errp_model_score(z, NULL);
        float ref = ERRP_GOLDEN_SCORE_INT8[i];
        float rel = fabsf(s - ref) / fmaxf(ref, 1e-6f);
        bool pass = rel < 0.02f && fabsf(lda - ERRP_GOLDEN_LDA[i]) < 1e-3f;
        ok &= pass;
        ESP_LOGI(TAG, "golden %d (label %d): S3 %.5f, PC int8 %.5f, float %.5f, err rel %.2e, LDA %.4f/%.4f %s"
                 " (%" PRId64 " us)", i, ERRP_GOLDEN_LABEL[i], s, ref, ERRP_GOLDEN_SCORE_FLOAT[i], rel, lda,
                 ERRP_GOLDEN_LDA[i], pass ? "OK" : "FALLA", errp_model_last_us());
    }
    printf("{\"ev\":\"selftest\",\"pass\":%d,\"arena\":%u,\"inf_us\":%" PRId64 "}\n", ok,
           (unsigned)errp_model_arena_used(), errp_model_last_us());
    return ok;
}

void detector_start(void)
{
    errp_ring_init(&s_ring);
    s_ring_lock = xSemaphoreCreateMutex();
    s_sync_q = xQueueCreate(8, sizeof(int64_t));

    calib_defaults();
    bool loaded = calib_load();
    ESP_LOGI(TAG, "calibración: %s", loaded ? "cargada de NVS" : "valores offline de errp_params.h");
    errp_alert_init(&s_alert, active_thresholds()[0], active_thresholds()[1], active_thresholds()[2]);
    s_alert.level = 3;  // paro seguro hasta calibrar, un epoch limpio <= T1 o CONFIRM

    gpio_config_t sync = {
        .pin_bit_mask = 1ULL << CONFIG_DETECTOR_SYNC_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_POSEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&sync));
    ESP_ERROR_CHECK(gpio_install_isr_service(ESP_INTR_FLAG_IRAM));
    ESP_ERROR_CHECK(gpio_isr_handler_add(CONFIG_DETECTOR_SYNC_GPIO, sync_isr, NULL));
#if CONFIG_DETECTOR_CALIB_BUTTON_GPIO >= 0
    gpio_config_t btn = {
        .pin_bit_mask = 1ULL << CONFIG_DETECTOR_CALIB_BUTTON_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&btn));
#endif

    eeg_input_start(&s_ring, s_ring_lock);
#if CONFIG_DETECTOR_CALIB_ON_BOOT
    calib_begin();
#endif
    xTaskCreatePinnedToCore(detector_task, "detector", 8192, NULL, 10, NULL, 1);
#if CONFIG_DETECTOR_USE_LDA
    const char *scorer = "LDA";
#else
    const char *scorer = "autoencoder";
#endif
    ESP_LOGI(TAG, "sync GPIO%d (flanco de subida), offset %d us, scorer %s",
             CONFIG_DETECTOR_SYNC_GPIO, CONFIG_DETECTOR_EVENT_LATENCY_OFFSET_US, scorer);
}
