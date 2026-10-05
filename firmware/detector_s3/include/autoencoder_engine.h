// Motor de inferencia del autoencoder del detector (ESP32-S3), en C nativo y float32.
// Sin librerías de inferencia (ni TFLite Micro, ni ESP-DL, ni ESP-DSP), sin heap y
// sin dependencias de ESP-IDF: el mismo código corre en el S3 y en el PC (tests).
//
// Modelo: autoencoder denso 40 -> 16 -> 6 -> 16 -> 40. Pesos, normalización,
// activaciones y umbrales vienen de autoencoder_weights.h, generado por
// ml/c_exporter/export_to_c.py (NO editar a mano; solo lo incluye autoencoder_engine.c).
//   z     = (x - MEAN_VECTOR) / STD_VECTOR                   (z-score, división float32)
//   h1    = act1(W1 z  + B1)    h2    = act2(W2 h1 + B2)     (W en layout nn.Linear [out][in])
//   h3    = act3(W3 h2 + B3)    z_hat = act4(W4 h3 + B4)
//   score = (1/40) * sum_i (z[i] - z_hat[i])^2               (MSE en el espacio NORMALIZADO)
// Los umbrales son percentiles de ese mismo score sobre datos normales. Un score
// igual a un umbral ya escala de nivel:
//   score >= T3 (p99.9) -> 3 SEVERE    parada segura
//   score >= T2 (p99)   -> 2 MODERATE  pausa / pedir confirmación
//   score >= T1 (p95)   -> 1 MILD      reducir velocidad
//   resto               -> 0 NORMAL    velocidad nominal
//
// Fail-safe (nunca operar a ciegas): si alguna x[i] o z[i] no es finita, o el score
// no es finito (NaN, Inf, desbordamiento), el resultado es valid = false,
// score = +INFINITY (nunca NaN: un "score > umbral" del llamador ve anomalía) y
// level = AE_LEVEL_SEVERE. features == NULL también cuenta como entrada inválida.
//
// Reentrante y thread-safe: sin estado global mutable, sin malloc ni printf.
// Pila: ~0.5 KB de buffers (z[40] + z_hat[40] en ae_infer_ex, h1[16] + h2[6] + h3[16]
// en ae_forward: 118 floats = 472 B). Medido en el S3 (GCC 15 -O2, -fstack-usage más
// los marcos de la libm), el peor caso con marcos de llamada es ~0.9 KB: ae_infer 48 +
// ae_infer_ex 400 + ae_forward ~200 + ae_activate 48 + erff/expf ~150-200 B. Dejar
// >= 1.5 KB libres en la tarea que llame a ae_infer.
// Coste: 1472 MACs por inferencia (40*16 + 16*6 + 6*16 + 16*40), más 40 divisiones
// de la normalización y las activaciones. Flash: 6.4 KB de pesos + ~2 KB de código.
//
// Integración ESP-IDF (src/CMakeLists.txt de firmware/detector_s3):
//   idf_component_register(SRCS "main.c" "autoencoder_engine.c"
//                          INCLUDE_DIRS "../include" "../../common")
// Uso:
//   float feats[AE_N_FEATURES];          // 40 features, en el orden del entrenamiento
//   ae_result_t r = ae_infer(feats);
//   if (!r.valid) { ... }                // dato inválido: r.level ya es AE_LEVEL_SEVERE
//   enviar_alerta(r.level);
// No compilar autoencoder_engine.c con -ffast-math / -ffinite-math-only: el fail-safe
// depende de NaN/Inf IEEE (el .c lo impide con #error).
//
// Regenerar pesos y vectores de test tras reentrenar:
//   python ml/c_exporter/export_to_c.py --model ae.pt --config config.json
// escribe include/autoencoder_weights.h y test/autoencoder_test_vectors.h (mismo
// AE_MODEL_ID); después recompilar y pasar test/test_c_engine.c.
#ifndef AUTOENCODER_ENGINE_H
#define AUTOENCODER_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tamaño del vector de features (entrada y salida del autoencoder)
#define AE_N_FEATURES 40

// Códigos de activación (autoencoder_weights.h los usa por nombre en LAYERk_ACT).
// param = LAYERk_ACT_PARAM; misma semántica que PyTorch.
#define AE_ACT_IDENTITY   0 // x
#define AE_ACT_RELU       1 // x > 0 ? x : 0 (NaN se propaga, como torch.relu)
#define AE_ACT_LEAKY_RELU 2 // x > 0 ? x : param * x       (param = negative_slope, def. 0.01)
#define AE_ACT_ELU        3 // x > 0 ? x : param * expm1(x) (param = alpha, def. 1.0)
#define AE_ACT_TANH       4 // tanh(x)
#define AE_ACT_SIGMOID    5 // 1 / (1 + exp(-x))
#define AE_ACT_SILU       6 // x * sigmoid(x)
#define AE_ACT_GELU       7 // 0.5 x (1 + erf(x / sqrt(2)))                     (approximate='none')
#define AE_ACT_GELU_TANH  8 // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))  (approximate='tanh')

typedef enum {
    AE_LEVEL_NORMAL = 0,   // velocidad nominal
    AE_LEVEL_MILD = 1,     // reducir velocidad
    AE_LEVEL_MODERATE = 2, // pausa / pedir confirmación
    AE_LEVEL_SEVERE = 3,   // parada segura (también cualquier dato inválido)
} ae_level_t;

typedef struct {
    float score;      // MSE normalizado; +INFINITY si !valid
    ae_level_t level; // nivel de alerta; AE_LEVEL_SEVERE si !valid
    bool valid;       // false: entrada no finita o score no finito (fail-safe)
} ae_result_t;

// Inferencia completa: normalizar, reconstruir, score y nivel.
// Equivale a ae_infer_ex(features, NULL, NULL).
ae_result_t ae_infer(const float features[AE_N_FEATURES]);

// Como ae_infer, devolviendo además z y z_hat (diagnóstico). z_out / z_hat_out pueden
// ser NULL (y pueden coincidir con features: se copian al final). Entrada inválida:
// ambos se llenan con NAN. Si solo el score desborda, contienen los valores calculados.
ae_result_t ae_infer_ex(const float features[AE_N_FEATURES], float z_out[AE_N_FEATURES],
                        float z_hat_out[AE_N_FEATURES]);

// z[i] = (x[i] - MEAN_VECTOR[i]) / STD_VECTOR[i]. Siempre escribe z; devuelve false si
// alguna x[i] o z[i] no es finita.
bool ae_normalize(const float x[AE_N_FEATURES], float z[AE_N_FEATURES]);

// Las 4 capas densas con sus activaciones, sin comprobaciones (z_hat no puede ser z).
void ae_forward(const float z[AE_N_FEATURES], float z_hat[AE_N_FEATURES]);

// (1/AE_N_FEATURES) * sum (z - z_hat)^2, en bruto (puede no ser finito).
float ae_score(const float z[AE_N_FEATURES], const float z_hat[AE_N_FEATURES]);

// Nivel según los umbrales (score == umbral escala). Score no finito -> SEVERE.
ae_level_t ae_level_from_score(float score);

// y = W x + b, con W row-major [n_out][n_in] (layout nn.Linear). y no puede ser x.
void ae_dense(const float *w, const float *b, const float *x, float *y, int n_in, int n_out);

// Activación in situ sobre v[0..n). Código desconocido -> v se llena con NAN
// (fail-safe: acaba en un resultado inválido / SEVERE).
void ae_activate(int act, float param, float *v, int n);

// Umbral a partir del cual empieza cada nivel: NORMAL -> 0.0f, MILD/MODERATE/SEVERE ->
// T1/T2/T3; cualquier otro valor -> +INFINITY.
float ae_threshold(ae_level_t level);

// Identificador del modelo compilado (AE_MODEL_ID: 16 hex del sha256 de pesos y umbrales).
const char *ae_model_id(void);

// true si los pesos son un placeholder/fixture aleatorio, NO un modelo entrenado.
bool ae_model_is_placeholder(void);

#ifdef __cplusplus
}
#endif

#endif // AUTOENCODER_ENGINE_H
