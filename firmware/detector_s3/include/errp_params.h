// GENERADO por ml/autoencoder/training/export_headers.py. No editar a mano.
// Datos de entrenamiento: synthetic_epochs.npz (SINTÉTICOS)
#pragma once
#include <stdint.h>

// Preprocesamiento (idéntico a ml/autoencoder/errp_pipeline.py)
// Filtro (en C1): scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos'), sosfilt causal
#define ERRP_FS_HZ         250
#define ERRP_N_CH          8
#define ERRP_PRE           50     // muestras de baseline, [-200, 0) ms
#define ERRP_POST          200    // muestras de [0, 800) ms
#define ERRP_DECIM         5     // media de cada 5 -> 50 Hz
#define ERRP_N_T           40    // muestras por canal que ve el modelo
#define ERRP_GATE_UV       100.0f
#define ERRP_FLAT_STD_UV   0.100000001f

// Normalización z-score por canal (offline; la calibración del S3 la reemplaza)
static const float ERRP_DEFAULT_MEAN[ERRP_N_CH] = {
    0.184472471f, 0.112320565f, -0.0194016732f, 0.0151566155f, 0.246424317f, 0.0842174813f, -0.233276233f, 0.345423728f
};
static const float ERRP_DEFAULT_STD[ERRP_N_CH] = {
    11.1833868f, 6.60497665f, 8.27284527f, 6.70692253f, 6.78470755f, 6.20486498f, 7.3311553f, 7.62548018f
};

// Umbrales del score MSE (offline; la calibración del S3 los reemplaza)
#define ERRP_DEFAULT_T1    0.759763241f  // p90
#define ERRP_DEFAULT_T2    1.24819207f  // p97
#define ERRP_DEFAULT_T3    1.47142708f  // p99
#define ERRP_PCT_T1        90.0f
#define ERRP_PCT_T2        97.0f
#define ERRP_PCT_T3        99.0f

// Baseline LDA (scorer de respaldo): score = w . f + b
// f[ch * 8 + bin] = media µV (baseline restado) en [edges[bin], edges[bin+1]) muestras desde t = 0
#define ERRP_LDA_N_BINS    8
static const int16_t ERRP_LDA_EDGES[ERRP_LDA_N_BINS + 1] = { 38, 55, 72, 89, 106, 123, 141, 158, 175 };
static const float ERRP_LDA_W[ERRP_N_CH * ERRP_LDA_N_BINS] = {
    0.0178424865f, -0.0255279411f, 0.00296955789f, 0.0318295695f, -0.0213701949f, -0.00870508235f, -0.0220798403f, 0.00830410048f,
    0.0496058986f, -0.0736903101f, 0.0300910082f, 0.00558415847f, 0.0195945017f, -0.0472808369f, -0.0430352464f, 0.00357849686f,
    0.0329892412f, -0.0637937859f, 0.0323722549f, 0.0839430839f, 0.0216496233f, -0.0405025631f, -0.024050381f, -0.0302184746f,
    0.0265778825f, -0.0281136148f, -0.0072502913f, 0.0101412823f, 0.039447166f, -0.00852330867f, -0.00484089879f, -0.00342038809f,
    -0.00750506762f, -0.0472186394f, 0.026197115f, 0.0467578731f, 0.0259143654f, -0.0284345169f, -0.0195789319f, -0.0351549946f,
    0.0302522015f, -0.0492152944f, 0.0098034367f, -0.00207910431f, -0.0186338797f, 0.0167266764f, -0.025149934f, -0.0148179401f,
    0.00809119549f, -0.00719538843f, -0.00238278252f, 0.0258908644f, 0.0133348955f, 0.0123990756f, -0.015544856f, -0.00588503852f,
    0.0151435938f, -0.0382614024f, -0.00523889624f, 0.00759933982f, -0.00114455156f, 0.00193472719f, 0.0233656205f, -0.0487803109f
};
#define ERRP_LDA_B         -2.03663588f
#define ERRP_LDA_AUC       0.814722896f

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
