// Prueba del DSP en el PC contra los valores de referencia de Python (errp_golden.h).
//   gcc -O2 -Wall -Wextra -Werror -Iinclude test/host_dsp_test.c src/errp_dsp.c -lm -o /tmp/dsp && /tmp/dsp
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "errp_dsp.h"
#include "errp_golden.h"

static int fails;

static void check(const char *name, int ok, const char *fmt, double v)
{
    printf("%s %-50s ", ok ? "OK  " : "FAIL", name);
    printf(fmt, v);
    printf("\n");
    fails += !ok;
}

static double max_err(const float a[ERRP_N_CH][ERRP_N_T], const float b[ERRP_N_CH][ERRP_N_T])
{
    double e = 0;
    for (int c = 0; c < ERRP_N_CH; c++)
        for (int t = 0; t < ERRP_N_T; t++)
            e = fmax(e, fabs(a[c][t] - b[c][t]));
    return e;
}

int main(void)
{
    // Preprocesado y LDA sobre la ventana de referencia
    float x[ERRP_N_CH][ERRP_N_T];
    errp_preprocess(ERRP_GOLDEN_WIN, x);
    check("preprocesado [8][250] -> [8][40] vs Python, µV", max_err(x, ERRP_GOLDEN_WIN_X) < 1e-4, "%.2e",
          max_err(x, ERRP_GOLDEN_WIN_X));
    float lda = errp_lda_score(x);
    check("LDA de la ventana vs Python", fabsf(lda - ERRP_GOLDEN_WIN_LDA) < 1e-3, "%.2e",
          fabs(lda - ERRP_GOLDEN_WIN_LDA));

    // Épocas de C4: normalización por canal y LDA
    double nerr = 0, lerr = 0;
    for (int g = 0; g < ERRP_N_GOLDEN; g++) {
        float z[ERRP_N_CH][ERRP_N_T];
        memcpy(z, ERRP_GOLDEN_X[g], sizeof(z));
        lerr = fmax(lerr, fabs(errp_lda_score(z) - ERRP_GOLDEN_LDA[g]));
        errp_normalize(z, ERRP_GOLDEN_MEAN[g], ERRP_GOLDEN_STD[g]);
        nerr = fmax(nerr, max_err(z, ERRP_GOLDEN_INPUT[g]));
    }
    check("normalización por canal de épocas de C4 vs Python", nerr < 1e-4, "%.2e", nerr);
    check("LDA de épocas de C4 vs Python", lerr < 1e-3, "%.2e", lerr);

    // Ring buffer: escribir la ventana como stream y recortarla por contador
    static errp_ring_t ring;
    static float win[ERRP_N_CH][ERRP_EPOCH_LEN];
    errp_ring_init(&ring);
    uint32_t t0 = 100000;
    for (int k = 0; k < ERRP_EPOCH_LEN; k++) {
        float s[ERRP_N_CH], gyr[3] = {1.0f, -2.0f, 0.5f};
        for (int c = 0; c < ERRP_N_CH; c++) s[c] = ERRP_GOLDEN_WIN[c][k];
        if (k == 100) gyr[1] = 18.0f;  // pico en el eje y: rango 20 °/s
        errp_ring_put(&ring, t0 - ERRP_PRE + k, s, gyr, 0);
    }
    errp_window_info_t info;
    errp_epoch_status_t st = errp_epoch_cut(&ring, t0, win, &info);
    check("ring: corte por contador == ventana", st == ERRP_EPOCH_OK && !memcmp(win, ERRP_GOLDEN_WIN, sizeof(win)),
          "%.0f", st);
    check("ring: gyro = rango pico a pico máx. (20 °/s)", fabsf(info.gyro_metric - 20.0f) < 1e-5f, "%.3f",
          info.gyro_metric);
    check("ring: época futura -> not_ready", errp_epoch_cut(&ring, t0 + 1, win, &info) == ERRP_EPOCH_NOT_READY,
          "%.0f", 0);

    // Gate en el orden de build_dataset.py
    errp_epoch_cut(&ring, t0, win, &info);
    errp_preprocess(win, x);
    check("gate: ventana limpia -> ok", errp_epoch_gate(&info, x, 30.0f) == ERRP_EPOCH_OK, "%.0f", 0);
    check("gate: gyro 20 > 15 -> gyro", errp_epoch_gate(&info, x, 15.0f) == ERRP_EPOCH_GYRO, "%.0f", 0);
    errp_window_info_t fi = info;
    fi.flags_or = ERRP_SF_FILTER_RESET;
    check("gate: FILTER_RESET -> flags", errp_epoch_gate(&fi, x, 30.0f) == ERRP_EPOCH_FLAGS, "%.0f", 0);
    float xa[ERRP_N_CH][ERRP_N_T];
    memcpy(xa, x, sizeof(xa));
    xa[3][20] = 101.0f;
    check("gate: |X| = 101 µV -> amplitude", errp_epoch_gate(&info, xa, 30.0f) == ERRP_EPOCH_AMPLITUDE, "%.0f", 0);
    memcpy(xa, x, sizeof(xa));
    for (int t = 0; t < ERRP_N_T; t++) xa[5][t] = 3.0f + 0.01f * (t & 1);  // std 0.005 µV
    check("gate: canal plano -> flat", errp_epoch_gate(&info, xa, 30.0f) == ERRP_EPOCH_FLAT, "%.0f", 0);

    float s0[ERRP_N_CH] = {0}, g0[3] = {0};
    errp_ring_put(&ring, t0 + 10, s0, g0, ERRP_SF_GAP);
    for (uint32_t k = t0 + 11; k < t0 + ERRP_POST + 11; k++) errp_ring_put(&ring, k, s0, g0, 0);
    st = errp_epoch_cut(&ring, t0 + 11, win, &info);
    check("ring: GAP dentro de la ventana se propaga en flags", st == ERRP_EPOCH_OK && (info.flags_or & ERRP_SF_GAP),
          "%.0f", info.flags_or);

    // IIR con offset DC grande y estado estacionario inicial
    errp_iir_t f;
    errp_iir_reset(&f, ERRP_GOLDEN_IIR_X[0]);
    double ierr = 0;
    for (int i = 0; i < ERRP_GOLDEN_IIR_LEN; i++) {
        ierr = fmax(ierr, fabs(errp_iir_step(&f, ERRP_GOLDEN_IIR_X[i]) - ERRP_GOLDEN_IIR_Y[i]));
    }
    check("IIR vs scipy sosfilt, µV (< 0.01)", ierr < 0.01, "%.2e", ierr);

    float v[] = {5, 1, 4, 2, 3};
    float p = errp_percentile(v, 5, 90.0f);
    check("percentil p90 de 1..5 == 4.6 (numpy)", fabsf(p - 4.6f) < 1e-5f, "%.4f", p);

    errp_alert_t a;
    errp_alert_init(&a, 1.0f, 2.0f, 3.0f);
    int l1 = errp_alert_score(&a, 1.5f);
    int l2 = errp_alert_score(&a, 2.5f);
    int l3 = errp_alert_score(&a, 2.5f);  // 2 seguidas > T2 -> 3
    int l4 = errp_alert_score(&a, 0.5f);
    errp_alert_rejected(&a);
    errp_alert_rejected(&a);
    int l5 = errp_alert_rejected(&a);     // 3 rechazadas -> 2
    int l6 = errp_alert_score(&a, 3.5f);
    check("alertas 1,2,3,0, rechazos->2, >T3->3",
          l1 == 1 && l2 == 2 && l3 == 3 && l4 == 0 && l5 == 2 && l6 == 3, "%.0f", 0);

    printf(fails ? "\n%d FALLAS\n" : "\ntodo OK\n", fails);
    return fails != 0;
}
