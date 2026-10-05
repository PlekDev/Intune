// Motor de inferencia del autoencoder (detector ESP32-S3): C puro en float32.
// Semántica, fail-safe, coste e integración: ver autoencoder_engine.h.
//
// Reglas de este archivo:
//   - Solo float32: todo literal lleva sufijo f y se usan expf/expm1f/tanhf/erff.
//     El S3 solo tiene FPU de simple precisión; cualquier double se emula por
//     software (lento). Compila limpio con -Wdouble-promotion -Wfloat-conversion.
//   - Sin malloc, sin printf, sin estado global mutable: reentrante. Todos los
//     buffers en la pila, dimensionados con las macros del header de pesos.
//   - Es el ÚNICO archivo que incluye autoencoder_weights.h (arrays static const):
//     así hay una sola copia de los pesos en flash.
#include "autoencoder_engine.h"
#include "autoencoder_weights.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

// El fail-safe depende de la semántica IEEE de NaN/Inf: con -ffast-math (o
// -ffinite-math-only) el compilador puede suponer que no existen, borrar los
// isfinite() y dejar pasar datos inválidos como NORMAL.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "autoencoder_engine.c no admite -ffast-math ni -ffinite-math-only: el fail-safe necesita NaN/Inf IEEE"
#endif

#if !defined(INPUT_DIM) || !defined(LAYER1_DIM) || !defined(LATENT_DIM) || !defined(LAYER3_DIM) ||    \
    !defined(OUTPUT_DIM) || !defined(THRESHOLD_LEVEL_1) || !defined(THRESHOLD_LEVEL_2) ||               \
    !defined(THRESHOLD_LEVEL_3) || !defined(LAYER1_ACT) || !defined(LAYER1_ACT_PARAM) ||                \
    !defined(LAYER2_ACT) || !defined(LAYER2_ACT_PARAM) || !defined(LAYER3_ACT) ||                       \
    !defined(LAYER3_ACT_PARAM) || !defined(LAYER4_ACT) || !defined(LAYER4_ACT_PARAM) ||                 \
    !defined(AE_MODEL_ID) || !defined(AE_WEIGHTS_PLACEHOLDER)
#error "autoencoder_weights.h incompleto o de otra version: regenerar con ml/c_exporter/export_to_c.py"
#endif

_Static_assert(INPUT_DIM == AE_N_FEATURES, "INPUT_DIM del modelo != AE_N_FEATURES: el modelo no encaja con las 40 features");
_Static_assert(OUTPUT_DIM == INPUT_DIM, "el autoencoder debe reconstruir su entrada (OUTPUT_DIM == INPUT_DIM)");
_Static_assert(sizeof(float) == 4, "se asume float IEEE-754 de 32 bits");

// Constantes de GELU (mismas que PyTorch: M_SQRT1_2 y sqrt(2/pi), redondeadas a float)
#define AE_INV_SQRT2      0.707106781f
#define AE_SQRT_2_OVER_PI 0.797884561f
#define AE_GELU_KAPPA     0.044715f

static void ae_fill_nan(float *v, int n)
{
    for (int i = 0; i < n; i++) {
        v[i] = NAN;
    }
}

static ae_result_t ae_invalid_result(void)
{
    // +INFINITY y no NaN: cualquier "score > umbral" del llamador ve una anomalía
    const ae_result_t r = {.score = INFINITY, .level = AE_LEVEL_SEVERE, .valid = false};
    return r;
}

void ae_dense(const float *w, const float *b, const float *x, float *y, int n_in, int n_out)
{
    // Producto escalar por fila con un solo acumulador float que parte del sesgo (mismo
    // orden de suma en todas las plataformas). En el S3 (gnu17 => -ffp-contract=fast)
    // GCC 15 -O2 deja el bucle interno en un loop sin overhead: lsi + lsi + madd.s (FMA).
    // Varios acumuladores acortarían la cadena de dependencias de madd.s, pero no
    // compensan la complejidad: la inferencia entera son 1472 MACs.
    const float *row = w;
    for (int o = 0; o < n_out; o++, row += n_in) {
        float acc = b[o];
        for (int i = 0; i < n_in; i++) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

void ae_activate(int act, float param, float *v, int n)
{
    // switch fuera del bucle: una sola decisión por capa, bucles simples por caso
    switch (act) {
    case AE_ACT_IDENTITY:
        break;
    case AE_ACT_RELU:
        for (int i = 0; i < n; i++) {
            // "v < 0 ? 0 : v" (y no "v > 0 ? v : 0") para que NaN se propague como en torch.relu
            v[i] = (v[i] < 0.0f) ? 0.0f : v[i];
        }
        break;
    case AE_ACT_LEAKY_RELU:
        for (int i = 0; i < n; i++) {
            v[i] = (v[i] > 0.0f) ? v[i] : param * v[i];
        }
        break;
    case AE_ACT_ELU:
        for (int i = 0; i < n; i++) {
            v[i] = (v[i] > 0.0f) ? v[i] : param * expm1f(v[i]);
        }
        break;
    case AE_ACT_TANH:
        for (int i = 0; i < n; i++) {
            v[i] = tanhf(v[i]);
        }
        break;
    case AE_ACT_SIGMOID:
        for (int i = 0; i < n; i++) {
            v[i] = 1.0f / (1.0f + expf(-v[i]));
        }
        break;
    case AE_ACT_SILU:
        for (int i = 0; i < n; i++) {
            // x * sigmoid(x) escrito como lo calcula PyTorch: x / (1 + exp(-x))
            v[i] = v[i] / (1.0f + expf(-v[i]));
        }
        break;
    case AE_ACT_GELU:
        for (int i = 0; i < n; i++) {
            v[i] = 0.5f * v[i] * (1.0f + erff(v[i] * AE_INV_SQRT2));
        }
        break;
    case AE_ACT_GELU_TANH:
        for (int i = 0; i < n; i++) {
            const float x = v[i];
            const float x3 = x * x * x;
            v[i] = 0.5f * x * (1.0f + tanhf(AE_SQRT_2_OVER_PI * (x + AE_GELU_KAPPA * x3)));
        }
        break;
    default:
        // Código desconocido (header corrupto o de otra versión): NaN -> resultado inválido
        ae_fill_nan(v, n);
        break;
    }
}

bool ae_normalize(const float x[AE_N_FEATURES], float z[AE_N_FEATURES])
{
    bool ok = true;
    for (int i = 0; i < AE_N_FEATURES; i++) {
        z[i] = (x[i] - MEAN_VECTOR[i]) / STD_VECTOR[i];
        if (!isfinite(x[i]) || !isfinite(z[i])) {
            ok = false;
        }
    }
    return ok;
}

void ae_forward(const float z[AE_N_FEATURES], float z_hat[AE_N_FEATURES])
{
    float h1[LAYER1_DIM];
    float h2[LATENT_DIM];
    float h3[LAYER3_DIM];

    // Los W son arrays 2D contiguos [out][in]: se recorren como row-major plano
    ae_dense((const float *)W1, B1, z, h1, INPUT_DIM, LAYER1_DIM);
    ae_activate(LAYER1_ACT, LAYER1_ACT_PARAM, h1, LAYER1_DIM);
    ae_dense((const float *)W2, B2, h1, h2, LAYER1_DIM, LATENT_DIM);
    ae_activate(LAYER2_ACT, LAYER2_ACT_PARAM, h2, LATENT_DIM);
    ae_dense((const float *)W3, B3, h2, h3, LATENT_DIM, LAYER3_DIM);
    ae_activate(LAYER3_ACT, LAYER3_ACT_PARAM, h3, LAYER3_DIM);
    ae_dense((const float *)W4, B4, h3, z_hat, LAYER3_DIM, OUTPUT_DIM);
    ae_activate(LAYER4_ACT, LAYER4_ACT_PARAM, z_hat, OUTPUT_DIM);
}

float ae_score(const float z[AE_N_FEATURES], const float z_hat[AE_N_FEATURES])
{
    float acc = 0.0f;
    for (int i = 0; i < AE_N_FEATURES; i++) {
        const float d = z[i] - z_hat[i];
        acc += d * d;
    }
    return acc / (float)AE_N_FEATURES;
}

ae_level_t ae_level_from_score(float score)
{
    // isfinite primero: con NaN todas las comparaciones son falsas y caería en NORMAL
    if (!isfinite(score) || score >= THRESHOLD_LEVEL_3) {
        return AE_LEVEL_SEVERE;
    }
    if (score >= THRESHOLD_LEVEL_2) {
        return AE_LEVEL_MODERATE;
    }
    if (score >= THRESHOLD_LEVEL_1) {
        return AE_LEVEL_MILD;
    }
    return AE_LEVEL_NORMAL;
}

ae_result_t ae_infer_ex(const float features[AE_N_FEATURES], float z_out[AE_N_FEATURES],
                        float z_hat_out[AE_N_FEATURES])
{
    float z[AE_N_FEATURES];
    float z_hat[AE_N_FEATURES];

    // Entrada inválida (NULL, NaN/Inf o z no finita): la red no se ejecuta
    if (features == NULL || !ae_normalize(features, z)) {
        if (z_out != NULL) {
            ae_fill_nan(z_out, AE_N_FEATURES);
        }
        if (z_hat_out != NULL) {
            ae_fill_nan(z_hat_out, AE_N_FEATURES);
        }
        return ae_invalid_result();
    }

    ae_forward(z, z_hat);
    const float score = ae_score(z, z_hat);

    // Copias al final: z_out / z_hat_out pueden apuntar a features
    if (z_out != NULL) {
        memcpy(z_out, z, sizeof(z));
    }
    if (z_hat_out != NULL) {
        memcpy(z_hat_out, z_hat, sizeof(z_hat));
    }

    // Desbordamiento (features enormes pero finitas) o NaN dentro de la red
    if (!isfinite(score)) {
        return ae_invalid_result();
    }
    const ae_result_t r = {.score = score, .level = ae_level_from_score(score), .valid = true};
    return r;
}

ae_result_t ae_infer(const float features[AE_N_FEATURES])
{
    return ae_infer_ex(features, NULL, NULL);
}

float ae_threshold(ae_level_t level)
{
    switch (level) {
    case AE_LEVEL_NORMAL:
        return 0.0f;
    case AE_LEVEL_MILD:
        return THRESHOLD_LEVEL_1;
    case AE_LEVEL_MODERATE:
        return THRESHOLD_LEVEL_2;
    case AE_LEVEL_SEVERE:
        return THRESHOLD_LEVEL_3;
    default:
        return INFINITY;
    }
}

const char *ae_model_id(void)
{
    return AE_MODEL_ID;
}

bool ae_model_is_placeholder(void)
{
    return AE_WEIGHTS_PLACEHOLDER != 0;
}
