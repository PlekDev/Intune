// ErrP-AE int8 con TFLite Micro (interfaz C).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "errp_params.h"

#ifdef __cplusplus
extern "C" {
#endif

bool errp_model_init(void);

// Epoch normalizado [8][40] -> score MSE (float, tras decuantizar). NAN si falla.
// recon (opcional) recibe la reconstrucción decuantizada.
float errp_model_score(const float e[ERRP_N_CH][ERRP_N_T], float recon[ERRP_N_CH][ERRP_N_T]);

size_t errp_model_arena_used(void);
int64_t errp_model_last_us(void);  // duración de la última inferencia

#ifdef __cplusplus
}
#endif
