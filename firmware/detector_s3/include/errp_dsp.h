// DSP del Detector (C2) en C puro, sin ESP-IDF: se prueba en el PC (test/host_dsp_test.c).
// Replica la puerta de ml/data/build_dataset.py (C4, FORMAT.md §3) y el LDA de baseline_lda.py.
// Preprocesado, normalización, inferencia y calibración del AE: autoencoder_engine.h (ae_*).
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "errp_params.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------- muestras y ring buffer indexado por contador Unicorn ----------
// Mismos bits que LINK_F_* de firmware/common/link_protocol.h
#define ERRP_SF_HELD          0x01
#define ERRP_SF_GAP           0x02
#define ERRP_SF_SETTLING      0x04
#define ERRP_SF_FILTER_RESET  0x08
#define ERRP_SF_UNFILTERED    0x20
// FORMAT.md §3 "flags" + "unfiltered": cualquiera en la ventana rechaza la época
#define ERRP_SF_REJECT (ERRP_SF_HELD | ERRP_SF_GAP | ERRP_SF_SETTLING | ERRP_SF_FILTER_RESET | ERRP_SF_UNFILTERED)

#define ERRP_RING_LEN 1024  // 4.1 s a 250 Hz (potencia de 2)

typedef struct {
    float eeg[ERRP_RING_LEN][ERRP_N_CH];  // µV filtrados
    float gyr[ERRP_RING_LEN][3];          // °/s
    uint32_t counter[ERRP_RING_LEN];      // contador real guardado (detecta sobrescritura)
    uint8_t flags[ERRP_RING_LEN];
    uint32_t newest;                      // último contador escrito
    bool any;
} errp_ring_t;

void errp_ring_init(errp_ring_t *r);
void errp_ring_put(errp_ring_t *r, uint32_t counter, const float eeg[ERRP_N_CH], const float gyr[3], uint8_t flags);

// ---------- época ----------
#define ERRP_EPOCH_LEN (ERRP_PRE + ERRP_POST)  // [-200, +800) ms = 250 muestras

typedef enum {
    ERRP_EPOCH_OK = 0,
    ERRP_EPOCH_NOT_READY,   // aún no llegan las muestras hasta +800 ms
    ERRP_EPOCH_COUNTER,     // muestras sobrescritas, nunca recibidas o contador no continuo
    ERRP_EPOCH_FLAGS,       // HELD / GAP / SETTLING / FILTER_RESET / UNFILTERED en la ventana
    ERRP_EPOCH_FLAT,        // algún canal de X con std < ERRP_FLAT_STD_UV
    ERRP_EPOCH_AMPLITUDE,   // max |X| > ERRP_GATE_UV
    ERRP_EPOCH_GYRO,        // rango pico a pico de gyro (máx. de 3 ejes) > umbral
} errp_epoch_status_t;

const char *errp_epoch_status_str(errp_epoch_status_t s);

typedef struct {
    uint8_t flags_or;     // OR de los flags de la ventana
    float gyro_metric;    // máx. sobre ejes del rango pico a pico, °/s
} errp_window_info_t;

// Copia [t0 - PRE, t0 + POST) del ring en win[ch][i]. Devuelve OK, NOT_READY o COUNTER.
errp_epoch_status_t errp_epoch_cut(const errp_ring_t *r, uint32_t t0_counter,
                                   float win[ERRP_N_CH][ERRP_EPOCH_LEN], errp_window_info_t *info);

// FORMAT.md §3 en el orden de build_dataset.py: flags, flat, amplitude, gyro.
// gate_gyro_dps <= 0 desactiva el gate de movimiento.
errp_epoch_status_t errp_epoch_gate(const errp_window_info_t *info, const float x[ERRP_N_CH][ERRP_N_T],
                                    float gate_gyro_dps);

// Score LDA (respaldo) sobre X [8][40] en µV (sin normalizar).
float errp_lda_score(const float x[ERRP_N_CH][ERRP_N_T]);

// Percentil con interpolación lineal (igual que numpy.percentile por defecto). Ordena v.
float errp_percentile(float *v, int n, float pct);

// ---------- IIR causal (solo modo provisional de tramas crudas) ----------
typedef struct {
    double z[ERRP_IIR_N_SOS][2];  // double: el offset DC del Unicorn crudo es de miles de µV
} errp_iir_t;

void errp_iir_reset(errp_iir_t *f, float x0);  // estado estacionario para x0 (sosfilt_zi * x0)
float errp_iir_step(errp_iir_t *f, float x);

// ---------- lógica de alertas (CLAUDE.md, Alert levels) ----------
typedef struct {
    float t1, t2, t3;
    int level;
    int gt2_streak;
    int rej_streak;
} errp_alert_t;

void errp_alert_init(errp_alert_t *a, float t1, float t2, float t3);
// Una época puntuada. Devuelve el nivel nuevo.
int errp_alert_score(errp_alert_t *a, float score);
// Una época rechazada: mantiene el nivel; 3 seguidas -> al menos 2.
int errp_alert_rejected(errp_alert_t *a);
// El operador confirmó la pausa en el brazo.
void errp_alert_confirm(errp_alert_t *a);

#ifdef __cplusplus
}
#endif
