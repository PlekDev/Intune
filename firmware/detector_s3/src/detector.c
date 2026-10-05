#include "detector.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "arm_link.h"
#include "autoencoder_engine.h"
#include "driver/gpio.h"
#include "eeg_input.h"
#include "errp_dsp.h"
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
#define NVS_KEY_AE   "ae_calib"  // ae_calib_t tal cual (incluye model_id)
#define NVS_KEY_LDA  "lda_t"     // float[3] umbrales del LDA

typedef struct {
    int64_t t_us;       // flanco de sincronía
    uint32_t counter;   // muestra t = 0
    bool counter_ok;
    bool overlap;
} pulse_t;

static errp_ring_t s_ring;
static SemaphoreHandle_t s_ring_lock;
static QueueHandle_t s_sync_q;
static ae_calib_t s_ae_cal;   // normalización + T1-T3 del AE + model_id (motor ae_*)
static float s_lda_t[3];      // T1-T3 del LDA (respaldo)
static ae_workspace_t s_ws;   // buffers de ae_infer (solo la tarea detector)
static int64_t s_inf_us;      // duración de la última ae_infer
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
    ae_calib_default(&s_ae_cal);  // normalización y umbrales del modelo cargado + model_id
    // Sin umbrales LDA de una sesión: nunca alerta hasta calibrar (con LDA activo, calibrar antes).
    s_lda_t[0] = s_lda_t[1] = s_lda_t[2] = INFINITY;
}

static bool calib_load(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK) {
        return false;
    }
    ae_calib_t c;
    float t[3];
    size_t len = sizeof(c), len_t = sizeof(t);
    // ae_calib_check rechaza blobs corruptos o de otro modelo (model_id distinto)
    bool ok = nvs_get_blob(h, NVS_KEY_AE, &c, &len) == ESP_OK && len == sizeof(c) && ae_calib_check(&c);
    if (ok) {
        s_ae_cal = c;
        if (nvs_get_blob(h, NVS_KEY_LDA, t, &len_t) == ESP_OK && len_t == sizeof(t)) {
            memcpy(s_lda_t, t, sizeof(t));
        }
    }
    nvs_close(h);
    return ok;
}

static void calib_save(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_blob(h, NVS_KEY_AE, &s_ae_cal, sizeof(s_ae_cal));
        nvs_set_blob(h, NVS_KEY_LDA, s_lda_t, sizeof(s_lda_t));
        nvs_commit(h);
        nvs_close(h);
    }
}

static const float *active_thresholds(void)
{
#if CONFIG_DETECTOR_USE_LDA
    return s_lda_t;
#else
    static float t[3];
    t[0] = s_ae_cal.t1;
    t[1] = s_ae_cal.t2;
    t[2] = s_ae_cal.t3;
    return t;
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
    // mean/std por canal (Welford, ddof 0) y T1-T3 = p90/p97/p99 de los scores, con el motor ae_*
    ae_calib_t c;
    ae_calib_default(&c);  // model_id + umbrales válidos mientras se calculan los scores
    ae_norm_acc_t acc;
    ae_norm_acc_init(&acc);
    for (int i = 0; i < s_calib_n; i++) {
        ae_norm_acc_add(&acc, (const float (*)[AE_N_T])s_calib_epochs[i]);
    }
    static float scores[150];
    int n = 0;
    bool ok = ae_norm_acc_finish(&acc, &c);
    for (int i = 0; ok && i < s_calib_n; i++) {
        ae_result_t r = ae_infer((const float (*)[AE_N_T])s_calib_epochs[i], &c, &s_ws);
        if (r.valid) {
            scores[n++] = r.score;
        }
    }
    ok = ok && ae_calib_thresholds_from_scores(&c, scores, n);
    s_calibrating = false;
    if (!ok) {
        ESP_LOGE(TAG, "calibración inválida (%d scores válidos de %d): se quedan los valores anteriores",
                 n, s_calib_n);
        printf("{\"ev\":\"calib_failed\",\"n\":%d,\"valid\":%d}\n", s_calib_n, n);
        return;
    }
    s_ae_cal = c;
    const float pct[3] = {ERRP_LDA_PCT_T1, ERRP_LDA_PCT_T2, ERRP_LDA_PCT_T3};
    for (int k = 0; k < 3; k++) {
        s_lda_t[k] = errp_percentile(s_calib_lda, s_calib_n, pct[k]);  // ordena s_calib_lda
    }
    calib_save();
    alert_reset_thresholds();
    s_alert.level = 0;  // se calibró con EEG limpio: arrancar en normal
    ESP_LOGW(TAG, "calibración lista (%d épocas): AE T1 %.4f T2 %.4f T3 %.4f | LDA T1 %.3f T2 %.3f T3 %.3f",
             s_calib_n, c.t1, c.t2, c.t3, s_lda_t[0], s_lda_t[1], s_lda_t[2]);
    printf("{\"ev\":\"calib_done\",\"n\":%d,\"t_ae\":[%.5f,%.5f,%.5f],\"t_lda\":[%.4f,%.4f,%.4f]}\n", s_calib_n,
           c.t1, c.t2, c.t3, s_lda_t[0], s_lda_t[1], s_lda_t[2]);
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
        ae_preprocess((const float (*)[AE_WIN_SAMPLES])win, e);  // X [8][40] µV, como el dataset de C4
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
            // normalizar -> int8 -> TFLM -> decuantizar -> MSE; inválido => score +inf (nivel 3)
            int64_t t0 = esp_timer_get_time();
            ae_result_t r = ae_infer((const float (*)[AE_N_T])e, &s_ae_cal, &s_ws);
            s_inf_us = esp_timer_get_time() - t0;
            ae = r.score;
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
           s_calibrating ? s_calib_n : -1, s_inf_us);
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
    // Los vectores dorados del motor (prueba 12) están en el entorno engine_test; aquí se
    // comprueba que el modelo cargó y se mide una inferencia con la calibración por defecto.
    const ae_model_info_t *mi = ae_model_info();
    ae_calib_t c;
    ae_calib_default(&c);
    static float e[AE_N_CH][AE_N_T];  // época de ceros: z = -mean/std, finita
    int64_t t0 = esp_timer_get_time();
    ae_result_t r = ae_infer((const float (*)[AE_N_T])e, &c, &s_ws);
    int64_t us = esp_timer_get_time() - t0;
    bool ok = ae_is_ready() && r.valid && us < 20000;
    ESP_LOGI(TAG, "modelo %s %s (%s), arena %lu / %lu B, inferencia %" PRId64 " us, score %.4f: %s",
             mi->arch, mi->model_id, mi->placeholder ? "PLACEHOLDER" : "entrenado",
             (unsigned long)mi->arena_used_bytes, (unsigned long)mi->arena_bytes, us, r.score, ok ? "OK" : "FALLA");
    printf("{\"ev\":\"selftest\",\"pass\":%d,\"model_id\":\"%s\",\"placeholder\":%d,\"arena\":%lu,\"inf_us\":%" PRId64
           "}\n", ok, mi->model_id, mi->placeholder, (unsigned long)mi->arena_used_bytes, us);
    return ok;
}

void detector_start(void)
{
    errp_ring_init(&s_ring);
    s_ring_lock = xSemaphoreCreateMutex();
    s_sync_q = xQueueCreate(8, sizeof(int64_t));

    calib_defaults();
    bool loaded = calib_load();
    ESP_LOGI(TAG, "calibración: %s", loaded ? "cargada de NVS (model_id coincide)"
                                          : "por defecto del modelo (autoencoder_weights.h)");
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
