// GENERADO por ml/autoencoder/training/export_headers.py. No editar a mano.
// Gate de synth_s1 (intune-c4-dataset-1.0); LDA de lda.json
#pragma once
#include <stdint.h>

// Cadena de FORMAT.md §2 (C4) / CLAUDE.md "Signal chain" (preprocesado: ae_preprocess)
#define ERRP_FS_HZ         250
#define ERRP_N_CH          8
#define ERRP_PRE           50     // muestras de baseline, [-200, 0) ms
#define ERRP_POST          200    // muestras de [0, 800) ms
#define ERRP_N_T           40    // muestras por canal de X (50 Hz)

// Puerta de artefactos (FORMAT.md §3), los valores con los que C4 construyó el dataset
#define ERRP_GATE_UV       100.0f   // max |X| tras baseline y diezmado
#define ERRP_FLAT_STD_UV   0.0500000007f   // std de un canal de X
#define ERRP_GATE_GYRO_DPS 30.0f   // rango pico a pico por eje (máx. de 3), ventana de 250

// Baseline LDA (scorer de respaldo): score = w . f + b
// f[ch * 8 + bin] = media de X[ch][edges[bin] .. edges[bin+1]) (µV, 50 Hz)
#define ERRP_LDA_N_BINS    8
static const int16_t ERRP_LDA_EDGES[ERRP_LDA_N_BINS + 1] = { 8, 11, 14, 18, 21, 25, 28, 32, 35 };
static const float ERRP_LDA_W[ERRP_N_CH * ERRP_LDA_N_BINS] = {
    -0.00234691333f, -0.0505763963f, 0.0382611938f, 0.0459871963f, -0.0172319505f, -0.00558901625f, 0.00422707573f, 0.0132211782f,
    -0.000945572159f, -0.0162428096f, 0.0193212088f, 0.00667204289f, -0.012350291f, -0.00894552283f, 0.0205455907f, -0.00691271154f,
    0.00379627198f, -0.0265862122f, 0.014414303f, 0.0250116643f, -0.0106921988f, -0.00914920587f, 0.00115499087f, -0.000784569944f,
    0.00277776923f, -0.0237570535f, 0.0206628088f, 0.0158437565f, 0.00111998396f, -0.000163853561f, -0.00327169057f, 0.00929629616f,
    -0.0117388936f, -0.0121894004f, 0.0126122087f, 0.0155991791f, -0.00456086174f, -0.00282106269f, 0.00274466048f, 0.0063887164f,
    0.00531753292f, 0.00884751417f, -0.0141966334f, -0.0076016169f, 0.0163040832f, -0.00334399287f, 0.0101550976f, -0.00451410841f,
    0.000842266774f, 0.00083063537f, -0.00207876088f, 0.00911189336f, 0.00325084431f, -0.00249023968f, 0.000620267005f, -0.00602992345f,
    0.0030363414f, -0.00800285116f, -0.00781121803f, -0.00354088959f, 0.000514525047f, -0.00429449975f, 0.000957066601f, 0.00199492672f
};
#define ERRP_LDA_B         -1.87975049f
#define ERRP_LDA_AUC       0.808258176f
#define ERRP_LDA_PCT_T1    90.0f
#define ERRP_LDA_PCT_T2    97.0f
#define ERRP_LDA_PCT_T3    99.0f

// IIR causal 1-15 Hz (regla dura 1). Lo aplica C1; el S3 lo usa SOLO en el modo
// provisional de tramas Unicorn crudas (DETECTOR_INPUT_RAW_UNICORN).
// Por sección: b0 b1 b2 a0 a1 a2 (a0 = 1). Estado inicial = ERRP_IIR_ZI * x0 (sosfilt_zi).
#define ERRP_IIR_N_SOS     2
static const float ERRP_IIR_SOS[ERRP_IIR_N_SOS][6] = {
    {
        0.024630608f, 0.0492612161f, 0.024630608f, 1.0f, -1.52773345f, 0.629663587f
    },
    {
        1.0f, -2.0f, 1.0f, 1.0f, -1.9650631f, 0.965762794f
    }
};
static const float ERRP_IIR_ZI[ERRP_IIR_N_SOS][2] = {
    {
        0.941937745f, -0.583982348f
    },
    {
        -0.966568351f, 0.966568351f
    }
};
