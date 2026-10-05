#include "errp_dsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void errp_ring_init(errp_ring_t *r)
{
    memset(r, 0, sizeof(*r));
}

void errp_ring_put(errp_ring_t *r, uint32_t counter, const float eeg[ERRP_N_CH], float gyr_max, uint8_t flags)
{
    uint32_t i = counter & (ERRP_RING_LEN - 1);
    memcpy(r->eeg[i], eeg, sizeof(r->eeg[i]));
    r->gyr_max[i] = gyr_max;
    r->counter[i] = counter;
    r->flags[i] = flags;
    r->newest = counter;
    r->any = true;
}

const char *errp_epoch_status_str(errp_epoch_status_t s)
{
    switch (s) {
    case ERRP_EPOCH_OK: return "ok";
    case ERRP_EPOCH_NOT_READY: return "not_ready";
    case ERRP_EPOCH_MISSING: return "missing";
    case ERRP_EPOCH_FLAGGED: return "flagged";
    case ERRP_EPOCH_AMPLITUDE: return "amplitude";
    case ERRP_EPOCH_MOTION: return "motion";
    case ERRP_EPOCH_FLAT: return "flat";
    }
    return "?";
}

errp_epoch_status_t errp_epoch_gate(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN])
{
    for (int c = 0; c < ERRP_N_CH; c++) {
        double s = 0, s2 = 0;
        for (int i = 0; i < ERRP_EPOCH_LEN; i++) {
            float v = ep[c][i];
            if (fabsf(v) > ERRP_GATE_UV) {
                return ERRP_EPOCH_AMPLITUDE;
            }
            s += v;
            s2 += (double)v * v;
        }
        double m = s / ERRP_EPOCH_LEN;
        double var = s2 / ERRP_EPOCH_LEN - m * m;
        if (var < (double)ERRP_FLAT_STD_UV * ERRP_FLAT_STD_UV) {
            return ERRP_EPOCH_FLAT;
        }
    }
    return ERRP_EPOCH_OK;
}

errp_epoch_status_t errp_epoch_cut(const errp_ring_t *r, uint32_t t0_counter, float gate_gyro_dps,
                                   float out[ERRP_N_CH][ERRP_EPOCH_LEN])
{
    uint32_t first = t0_counter - ERRP_PRE;
    uint32_t last = t0_counter + ERRP_POST - 1;
    if (!r->any || (int32_t)(r->newest - last) < 0) {
        return ERRP_EPOCH_NOT_READY;
    }
    if (r->newest - first >= ERRP_RING_LEN) {
        return ERRP_EPOCH_MISSING;  // ya sobrescrito
    }
    errp_epoch_status_t st = ERRP_EPOCH_OK;
    for (int k = 0; k < ERRP_EPOCH_LEN; k++) {
        uint32_t cnt = first + k;
        uint32_t i = cnt & (ERRP_RING_LEN - 1);
        if (r->counter[i] != cnt) {
            return ERRP_EPOCH_MISSING;
        }
        if (r->flags[i] & ERRP_SF_BAD) {
            st = ERRP_EPOCH_FLAGGED;
        } else if (st == ERRP_EPOCH_OK && gate_gyro_dps > 0 && r->gyr_max[i] > gate_gyro_dps) {
            st = ERRP_EPOCH_MOTION;
        }
        for (int c = 0; c < ERRP_N_CH; c++) {
            out[c][k] = r->eeg[i][c];
        }
    }
    return st != ERRP_EPOCH_OK ? st : errp_epoch_gate(out);
}

void errp_preprocess(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN], float out[ERRP_N_CH][ERRP_N_T])
{
    for (int c = 0; c < ERRP_N_CH; c++) {
        double base = 0;
        for (int i = 0; i < ERRP_PRE; i++) {
            base += ep[c][i];
        }
        float b = (float)(base / ERRP_PRE);
        for (int t = 0; t < ERRP_N_T; t++) {
            float s = 0;
            for (int k = 0; k < ERRP_DECIM; k++) {
                s += ep[c][ERRP_PRE + t * ERRP_DECIM + k] - b;
            }
            out[c][t] = s / ERRP_DECIM;
        }
    }
}

void errp_normalize(float e[ERRP_N_CH][ERRP_N_T], const float mean[ERRP_N_CH], const float std[ERRP_N_CH])
{
    for (int c = 0; c < ERRP_N_CH; c++) {
        for (int t = 0; t < ERRP_N_T; t++) {
            e[c][t] = (e[c][t] - mean[c]) / std[c];
        }
    }
}

float errp_lda_score(const float ep[ERRP_N_CH][ERRP_EPOCH_LEN])
{
    float score = ERRP_LDA_B;
    for (int c = 0; c < ERRP_N_CH; c++) {
        double base = 0;
        for (int i = 0; i < ERRP_PRE; i++) {
            base += ep[c][i];
        }
        float b = (float)(base / ERRP_PRE);
        for (int k = 0; k < ERRP_LDA_N_BINS; k++) {
            int a = ERRP_LDA_EDGES[k], z = ERRP_LDA_EDGES[k + 1];
            float s = 0;
            for (int i = a; i < z; i++) {
                s += ep[c][ERRP_PRE + i] - b;
            }
            score += ERRP_LDA_W[c * ERRP_LDA_N_BINS + k] * (s / (z - a));
        }
    }
    return score;
}

float errp_mse(const float a[ERRP_N_CH][ERRP_N_T], const float b[ERRP_N_CH][ERRP_N_T])
{
    double s = 0;
    for (int c = 0; c < ERRP_N_CH; c++) {
        for (int t = 0; t < ERRP_N_T; t++) {
            double d = (double)a[c][t] - b[c][t];
            s += d * d;
        }
    }
    return (float)(s / (ERRP_N_CH * ERRP_N_T));
}

static int cmp_float(const void *x, const void *y)
{
    float a = *(const float *)x, b = *(const float *)y;
    return (a > b) - (a < b);
}

float errp_percentile(float *v, int n, float pct)
{
    if (n <= 0) {
        return NAN;
    }
    qsort(v, n, sizeof(float), cmp_float);
    double pos = (double)pct / 100.0 * (n - 1);
    int lo = (int)floor(pos);
    int hi = lo + 1 < n ? lo + 1 : lo;
    double frac = pos - lo;
    return (float)(v[lo] + (v[hi] - v[lo]) * frac);
}

void errp_iir_reset(errp_iir_t *f, float x0)
{
    // zi de scipy está definido para la cascada: la entrada de la sección s es la
    // salida en estado estacionario de las anteriores; sosfilt_zi ya lo incorpora.
    for (int s = 0; s < ERRP_IIR_N_SOS; s++) {
        f->z[s][0] = (double)ERRP_IIR_ZI[s][0] * x0;
        f->z[s][1] = (double)ERRP_IIR_ZI[s][1] * x0;
    }
}

float errp_iir_step(errp_iir_t *f, float x)
{
    double v = x;
    for (int s = 0; s < ERRP_IIR_N_SOS; s++) {  // forma directa II transpuesta (como scipy)
        const float *c = ERRP_IIR_SOS[s];
        double y = c[0] * v + f->z[s][0];
        f->z[s][0] = c[1] * v - c[4] * y + f->z[s][1];
        f->z[s][1] = c[2] * v - c[5] * y;
        v = y;
    }
    return (float)v;
}

void errp_alert_init(errp_alert_t *a, float t1, float t2, float t3)
{
    memset(a, 0, sizeof(*a));
    a->t1 = t1;
    a->t2 = t2;
    a->t3 = t3;
}

int errp_alert_score(errp_alert_t *a, float s)
{
    a->rej_streak = 0;
    if (s > a->t3) {
        a->level = 3;
    } else if (s > a->t2) {
        a->level = a->gt2_streak >= 1 ? 3 : 2;
    } else if (s > a->t1) {
        a->level = 1;
    } else {
        a->level = 0;
    }
    a->gt2_streak = s > a->t2 ? a->gt2_streak + 1 : 0;
    return a->level;
}

int errp_alert_rejected(errp_alert_t *a)
{
    if (++a->rej_streak >= 3 && a->level < 2) {
        a->level = 2;  // operador no observable
    }
    return a->level;
}

void errp_alert_confirm(errp_alert_t *a)
{
    a->level = 0;
    a->gt2_streak = 0;
    a->rej_streak = 0;
}
