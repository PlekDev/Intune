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
    printf("%s %-42s ", ok ? "OK  " : "FAIL", name);
    printf(fmt, v);
    printf("\n");
    fails += !ok;
}

int main(void)
{
    // Epoch crudo de referencia: [-220, +820) ms -> ventana [-200, +800)
    static float ep[ERRP_N_CH][ERRP_EPOCH_LEN];
    int off = ERRP_GOLDEN_RAW_T0 - ERRP_PRE;
    for (int c = 0; c < ERRP_N_CH; c++) {
        memcpy(ep[c], &ERRP_GOLDEN_RAW[c][off], sizeof(ep[c]));
    }

    float pre[ERRP_N_CH][ERRP_N_T];
    errp_preprocess(ep, pre);
    double err = 0;
    for (int c = 0; c < ERRP_N_CH; c++)
        for (int t = 0; t < ERRP_N_T; t++)
            err = fmax(err, fabs(pre[c][t] - ERRP_GOLDEN_RAW_PRE[c][t]));
    check("preprocesado [8][40] vs Python, err máx µV", err < 1e-4, "%.2e", err);

    float lda = errp_lda_score(ep);
    check("score LDA vs Python, err abs", fabs(lda - ERRP_GOLDEN_RAW_LDA) < 1e-3, "%.2e", fabs(lda - ERRP_GOLDEN_RAW_LDA));
    check("gate acepta el epoch de referencia", errp_epoch_gate(ep) == ERRP_EPOCH_OK, "%.0f", 0);

    // Ring buffer: escribir el epoch como stream y recortarlo por contador
    static errp_ring_t ring;
    errp_ring_init(&ring);
    uint32_t t0 = 100000;
    for (int k = 0; k < ERRP_EPOCH_LEN; k++) {
        float s[ERRP_N_CH];
        for (int c = 0; c < ERRP_N_CH; c++) s[c] = ep[c][k];
        errp_ring_put(&ring, t0 - ERRP_PRE + k, s, 0, 0);
    }
    static float cut[ERRP_N_CH][ERRP_EPOCH_LEN];
    errp_epoch_status_t st = errp_epoch_cut(&ring, t0, 30.0f, cut);
    check("ring: corte por contador == epoch", st == ERRP_EPOCH_OK && !memcmp(cut, ep, sizeof(ep)), "%.0f", st);
    check("ring: epoch futuro -> not_ready", errp_epoch_cut(&ring, t0 + 1, 30.0f, cut) == ERRP_EPOCH_NOT_READY, "%.0f", 0);
    float s0[ERRP_N_CH] = {0};
    errp_ring_put(&ring, t0 + 10, s0, 0, ERRP_SF_GAP);  // nueva muestra marcada
    for (uint32_t k = t0 + 11; k < t0 + ERRP_POST + 11; k++) errp_ring_put(&ring, k, s0, 0, 0);
    st = errp_epoch_cut(&ring, t0 + 11, 30.0f, cut);
    check("ring: GAP dentro de la ventana -> flagged", st == ERRP_EPOCH_FLAGGED, "%.0f", st);

    // IIR con offset DC grande y estado estacionario inicial
    errp_iir_t f;
    errp_iir_reset(&f, ERRP_GOLDEN_IIR_X[0]);
    double ierr = 0;
    for (int i = 0; i < ERRP_GOLDEN_IIR_LEN; i++) {
        float y = errp_iir_step(&f, ERRP_GOLDEN_IIR_X[i]);
        ierr = fmax(ierr, fabs(y - ERRP_GOLDEN_IIR_Y[i]));
    }
    check("IIR vs scipy sosfilt, err máx µV (< 0.01)", ierr < 0.01, "%.2e", ierr);

    // Percentil como numpy (interpolación lineal)
    float v[] = {5, 1, 4, 2, 3};
    float p = errp_percentile(v, 5, 90.0f);
    check("percentil p90 de 1..5 == 4.6", fabsf(p - 4.6f) < 1e-5f, "%.4f", p);

    // Máquina de alertas
    errp_alert_t a;
    errp_alert_init(&a, 1.0f, 2.0f, 3.0f);
    int l1 = errp_alert_score(&a, 1.5f);
    int l2 = errp_alert_score(&a, 2.5f);
    int l3 = errp_alert_score(&a, 2.5f);  // 2 seguidos > T2 -> 3
    int l4 = errp_alert_score(&a, 0.5f);
    errp_alert_rejected(&a);
    errp_alert_rejected(&a);
    int l5 = errp_alert_rejected(&a);     // 3 rechazados -> 2
    int l6 = errp_alert_score(&a, 3.5f);
    check("alertas 1,2,3,0, rechazos->2, >T3->3",
          l1 == 1 && l2 == 2 && l3 == 3 && l4 == 0 && l5 == 2 && l6 == 3, "%.0f", 0);

    printf(fails ? "\n%d FALLAS\n" : "\ntodo OK\n", fails);
    return fails != 0;
}
