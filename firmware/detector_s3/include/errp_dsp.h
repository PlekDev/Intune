// DSP del Detector (C2) en C puro, sin ESP-IDF: se prueba en el PC (test/host_dsp_test.c).
// Replica ml/autoencoder/errp_pipeline.py paso a paso.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "errp_params.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------- muestras y ring buffer indexado por contador Unicorn ----------
#define ERRP_SF_HELD      0x01  // muestra repetida (1 faltante)
#define ERRP_SF_GAP       0x02  // parte de un hueco >= 2
#define ERRP_SF_SETTLING  0x04  // filtro reiniciado hace < 2 s
#define ERRP_SF_BAD       (ERRP_SF_HELD | ERRP_SF_GAP | ERRP_SF_SETTLING)

#define ERRP_RING_LEN 1024  // 4.1 s a 250 Hz (potencia de 2)

typedef struct {
    float eeg[ERRP_RING_LEN][ERRP_N_CH];  // µV filtrados
    float gyr_max[ERRP_RING_LEN];         // max |gyro| de la muestra, °/s
    uint32_t counter[ERRP_RING_LEN];      // contador real guardado (detecta sobrescritura)
    uint8_t flags[ERRP_RING_LEN];
    uint32_t newest;                      // último contador escrito
    bool any;
} errp_ring_t;

void errp_ring_init(errp_ring_t *r);
void errp_ring_put(errp_ring_t *r, uint32_t counter, const float eeg[ERRP_N_CH], float gyr_max, uint8_t flags);

// ---------- epoch ----------
#define ERRP_EPOCH_LEN (ERRP_PRE + ERRP_POST)  // [-200, +800) ms = 250 muestras

typedef enum {
    ERRP_EPOCH_OK = 0,
    ERRP_EPOCH_NOT_READY,   // aún no llegan las muestras hasta +800 ms
    ERRP_EPOCH_MISSING,     // muestras sobrescritas o nunca recibidas
    ERRP_EPOCH_FLAGGED,     // HELD / GAP / SETTLING dentro de la ventana
    ERRP_EPOCH_AMPLITUDE,   // pico > GATE_UV
    ERRP_EPOCH_MOTION,      // gyro > umbral
    ERRP_EPOCH_FLAT,        // canal plano
} errp_epoch_status_t;

const char *errp_epoch_status_str(errp_epoch_status_t s);

// Copia [t0 - PRE, t0 + POST) del ring en out[ch][i] y aplica el gate.
errp_epoch_status_t errp_epoch_cut(const errp_ring_t *r, uint32_t t0_counter, float gate_gyro_dps,
                                   float out[ERRP_N_CH][ERRP_EPOCH_LEN]);

// Gate de amplitud y canal plano sobre un epoch [8][250] (también lo usa la prueba en PC).
errp_epoch_status_t errp_epoch_gate(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN]);

// Baseline [-200, 0) restado y media de cada 5 sobre [0, 800): [8][250] -> [8][40].
void errp_preprocess(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN], float out[ERRP_N_CH][ERRP_N_T]);

void errp_normalize(float e[ERRP_N_CH][ERRP_N_T], const float mean[ERRP_N_CH], const float std[ERRP_N_CH]);

// Score LDA (respaldo) sobre el epoch [8][250] crudo filtrado.
float errp_lda_score(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN]);

// MSE entre epoch normalizado y su reconstrucción, ambos [8][40].
float errp_mse(const float a[ERRP_N_CH][ERRP_N_T], const float b[ERRP_N_CH][ERRP_N_T]);

// Percentil con interpolación lineal (igual que numpy.percentile por defecto). Ordena v.
float errp_percentile(float *v, int n, float pct);

// ---------- IIR causal (modo provisional de tramas crudas) ----------
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
// Un epoch puntuado. Devuelve el nivel nuevo.
int errp_alert_score(errp_alert_t *a, float score);
// Un epoch rechazado: mantiene el nivel; 3 seguidos -> al menos 2.
int errp_alert_rejected(errp_alert_t *a);
// El operador confirmó la pausa en el brazo.
void errp_alert_confirm(errp_alert_t *a);

#ifdef __cplusplus
}
#endif
