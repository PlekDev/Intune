// GENERADO por ml/autoencoder/training/export_headers.py. No editar a mano.
// Datos de entrenamiento: synthetic_epochs.npz (SINTÉTICOS)
#pragma once
#include <stddef.h>
#include <stdint.h>

#define ERRP_MODEL_ARCH        "dense"
#define ERRP_MODEL_PARAMS      43472
#define ERRP_MODEL_SYNTHETIC   1
// Ops: FULLY_CONNECTED, RESHAPE

// Cuantización (entrada y salida int8, forma [1, 8, 40, 1] NHWC)
#define ERRP_IN_SCALE          0.0731530413f
#define ERRP_IN_ZERO_POINT     -7
#define ERRP_OUT_SCALE         0.0652361512f
#define ERRP_OUT_ZERO_POINT    10
#define ERRP_FLOAT_INT8_CORR   0.999835372f

extern const uint8_t g_errp_model[];
extern const size_t g_errp_model_len;
