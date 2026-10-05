// GENERADO por ml/autoencoder/training/export_headers.py. No editar a mano.
// Datos: synth_s1, synth_s2, synth_s3, synth_s4 (SINTÉTICOS); dataset intune-c4-dataset-1.0
#pragma once
#include <stddef.h>
#include <stdint.h>

#define ERRP_MODEL_ARCH        "dense"
#define ERRP_MODEL_PARAMS      43472
#define ERRP_MODEL_SYNTHETIC   1
// Ops: FULLY_CONNECTED, RESHAPE

// Cuantización (entrada y salida int8, forma [1, 8, 40, 1] NHWC)
#define ERRP_IN_SCALE          0.0627036169f
#define ERRP_IN_ZERO_POINT     2
#define ERRP_OUT_SCALE         0.0593033507f
#define ERRP_OUT_ZERO_POINT    1
#define ERRP_FLOAT_INT8_CORR   0.999971688f

extern const uint8_t g_errp_model[];
extern const size_t g_errp_model_len;
