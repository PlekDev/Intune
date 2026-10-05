// GENERADO por tools/linux_probe/design_iir.py. No editar a mano.
// Band-pass causal Butterworth 1–15 Hz, orden 4, fs = 250 Hz.
// scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos')
// Retardo de grupo: 2 Hz 86 ms, 4 Hz 31 ms, 6 Hz 24 ms, 8 Hz 23 ms, 10 Hz 22 ms, 12 Hz 21 ms.
//
// Forma directa II transpuesta, igual que scipy.signal.sosfilt, en double: la señal
// cruda trae offsets de cientos de miles de µV y el pasa-altos de 1 Hz tiene polos
// cerca del círculo unidad; en float el error de redondeo se nota.
// eeg_iir_reset(x0) arranca en estado estacionario para x0 (= sosfilt_zi(sos) * x0),
// así el offset no provoca un transitorio de varios segundos.
#pragma once

#define EEG_IIR_N_SEC 2
#define EEG_IIR_SETTLE_SAMPLES 500  // 2 s marcado SETTLING tras cada reinicio (regla dura 3)

// Cada fila: b0 b1 b2 a0 a1 a2 (a0 = 1)
static const double EEG_IIR_SOS[EEG_IIR_N_SEC][6] = {
    {0.02463060804906882, 0.04926121609813764, 0.02463060804906882, 1, -1.52773348395035, 0.62966360587227577},
    {1, -2, 1, 1, -1.9650630430819829, 0.9657628068672337}
};

// Estado estacionario por sección para entrada constante 1
static const double EEG_IIR_ZI[EEG_IIR_N_SEC][2] = {
    {0.94193776583887157, -0.58398231957531388},
    {-0.96656837388794048, 0.96656837388794048}
};

typedef struct {
    double z[EEG_IIR_N_SEC][2];
} eeg_iir_t;

static inline void eeg_iir_reset(eeg_iir_t *f, double x0)
{
    for (int s = 0; s < EEG_IIR_N_SEC; s++) {
        f->z[s][0] = EEG_IIR_ZI[s][0] * x0;
        f->z[s][1] = EEG_IIR_ZI[s][1] * x0;
    }
}

static inline double eeg_iir_step(eeg_iir_t *f, double x)
{
    for (int s = 0; s < EEG_IIR_N_SEC; s++) {
        const double *c = EEG_IIR_SOS[s];
        double y = c[0] * x + f->z[s][0];
        f->z[s][0] = c[1] * x - c[4] * y + f->z[s][1];
        f->z[s][1] = c[2] * x - c[5] * y;
        x = y;
    }
    return x;
}
