// Motor del ErrP-AE del detector (C2, ESP32-S3): autoencoder convolucional int8 en
// TensorFlow Lite Micro (CLAUDE.md del equipo) más el pre/post-procesado, la calibración
// por operador y el fail-safe en C puro float32. CONTRACT v3, secciones 2 y 7.
//
// Cadena por época (la réplica offline bit a bit es ml/autoencoder/errp_ae.py):
//   win[8][250]   ventana a 250 Hz alrededor del sync corregido; muestra j -> t = (j - 50) * 4 ms
//                 (de -200 a +796 ms). Canales en orden Unicorn: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
//   ae_preprocess base[c] = media(win[c][0..49]) (línea base [-200, 0) ms);
//                 epoch[c][k] = media(win[c][50+5k .. 54+5k]) - base[c] (diezmado x5 por media,
//                 250 -> 50 Hz): época 8 x 40 (0-800 ms). Sumas float32 en orden de índice,
//                 media = suma / cuenta.
//   (puerta de artefactos: lógica del firmware, fuera de este módulo)
//   ae_normalize  z[c*40 + t] = (epoch[c][t] - mean[c]) / std[c]   (calibración del operador)
//   ae_quantize   q_in = clamp(rint(z / s_in) + zp_in, -128, 127)  (redondeo a par, modo FP por defecto)
//   ErrP-AE       q_out = modelo int8 [1, 8, 40, 1] -> [1, 8, 40, 1] (ae_runner_invoke, TFLM)
//   ae_dequantize z_hat = s_out * (float)(q_out - zp_out)
//   ae_score      score = (1/320) * sum (z - z_hat)^2   (MSE float32, orden de índice)
//   ae_level_from_score  0 si score <= T1; 1 si T1 < score <= T2; 2 si T2 < score <= T3;
//                 3 si score > T3 (">" estricto).
// ae_infer encadena normalize -> quantize -> invoke -> dequantize -> score -> nivel. Rachas,
// EPOCH_REJECTED, confirmación y demás fuentes de fail-safe son lógica de alertas del firmware.
// Época, z, q_in y z_hat (dado q_out) son bit a bit con la referencia Python; con FMA (madd.s en
// el S3, -ffp-contract) el score puede diferir ~1 ulp. q_out es bit a bit con los kernels de
// referencia de TFLM; tf.lite BUILTIN_REF (TF 2.21) recuantiza FULLY_CONNECTED con otro redondeo
// y difiere 1 LSB en los empates (por eso los goldens del test replican la aritmética de TFLM:
// export_to_c.py --golden-runtime tflm, el valor por defecto).
//
// Fail-safe (nunca operar a ciegas): época, z o score no finitos; calibración inválida (no
// finita, std <= 0, no 0 < t1 <= t2 <= t3, model_id de otro modelo); motor sin inicializar;
// fallo de TFLM al iniciar o invocar; punteros NULL -> valid = false, score = +INFINITY (nunca
// NaN: un "score > umbral" del llamador ve anomalía) y level = AE_LEVEL_SEVERE.
//
// Calibración por operador y sesión (modo calibración, ~60-100 acciones correctas; repetir en
// cada sesión). Usar épocas correctas que no se usaron para entrenar el modelo:
//   ae_calib_t cal;
//   ae_calib_default(&cal);                       // valores por defecto + model_id del modelo
//   ae_norm_acc_t acc;
//   ae_norm_acc_init(&acc);
//   for (...) ae_norm_acc_add(&acc, epoch_i);     // mean/std: Welford float32, ddof 0
//   if (!ae_norm_acc_finish(&acc, &cal)) { ... }  // falla -> cal.mean/std = NaN (inutilizable)
//   for (...) { ae_result_t r = ae_infer(epoch_j, &cal, &ws); if (r.valid) scores[n++] = r.score; }
//   if (!ae_calib_thresholds_from_scores(&cal, scores, n)) { ... }  // p90/p97/p99, n >= 20
//   nvs_set_blob(h, "ae_calib", &cal, sizeof(cal));
// Los scores necesitan la mean/std final: guardar las épocas (8 x 40 float = 1.25 KB cada una,
// mejor en PSRAM) o hacer dos fases (primeras épocas -> mean/std, siguientes -> scores).
// NVS: el blob es ae_calib_t tal cual (incluye model_id). Al arrancar, cargarlo y validarlo con
// ae_calib_check: tras cambiar de modelo (otro AE_MODEL_ID) o con un blob corrupto devuelve
// false y hay que recalibrar (o usar ae_calib_default de forma explícita). Comprobar también
// que el tamaño leído es sizeof(ae_calib_t).
//
// Memoria: modelo int8 en flash (AE_MODEL_TFLITE, ~50-70 KB, solo en autoencoder_tflm.cc; el
// runner añade ~19 KB de código, ~17 de ellos del verificador de flatbuffers); arena TFLM
// estática de AE_TENSOR_ARENA_BYTES (.bss, en el runner); ae_workspace_t (~3.2 KB) lo aporta el
// llamador (estático o en su pila); sin malloc. AE_TENSOR_ARENA_BYTES = arena medida en el PC
// (punteros de 64 bits: cota superior de la del S3) + margen; en el S3 la arena real queda en
// ae_model_info()->arena_used_bytes y, si no cabe (p.ej. scratch de ESP-NN), AllocateTensors
// falla y ae_init devuelve false. Pila (medida con -fstack-usage en el S3): ae_infer 112 B +
// normalize 64 + quantize 48 + runner 32, más la de Invoke de TFLM; dejar >= 4 KB libres en la
// tarea que llama (más el workspace si va en la pila). Coste: ~75k MACs int8 (modelo conv) por
// época; medir con el benchmark de test/test_c_engine.c (CLAUDE.md: < 20 ms por época).
//
// NO reentrante: hay un único intérprete TFLM (estado estático en autoencoder_tflm.cc) y el
// estado de ae_init (s_ready + info). Llamar a ae_init una vez al arrancar y a ae_infer siempre
// desde la misma tarea (o serializar con un mutex). Las funciones puras (preprocess, normalize,
// quantize, dequantize, score, level, percentiles, acumulador) sí son reentrantes.
//
// Integración ESP-IDF (componente de firmware/detector_s3):
//   idf.py add-dependency "espressif/esp-tflite-micro"    (escribe idf_component.yml:
//                                                           espressif/esp-tflite-micro: "^1.4.1")
//   idf_component_register(SRCS "autoencoder_engine.c" "autoencoder_tflm.cc" ...
//                          INCLUDE_DIRS "../include")
// esp-tflite-micro usa los kernels de ESP-NN en el S3: verificar en placa (test/test_c_engine.c,
// nivel C) que q_out sigue siendo bit a bit igual que los goldens de referencia.
// No compilar autoencoder_engine.c con -ffast-math / -ffinite-math-only (el .c lo impide con
// #error): el fail-safe depende de NaN/Inf IEEE.
// En C anterior a C23 con -Wpedantic, pasar un float[8][40] NO const a un parámetro
// "const float x[AE_N_CH][AE_N_T]" avisa: usar un cast (const float (*)[AE_N_T]).
//
// Regenerar el modelo tras reentrenar (ml/autoencoder/train_errp_ae.py):
//   python ml/c_exporter/export_to_c.py --model model.keras --config config.json --rep rep.npz --val val.npz
// escribe include/autoencoder_weights.h y test/autoencoder_test_vectors.h (mismo AE_MODEL_ID);
// después recompilar y pasar test/test_c_engine.c.
#ifndef AUTOENCODER_ENGINE_H
#define AUTOENCODER_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AE_N_CH 8               // electrodos (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
#define AE_N_T 40               // muestras por canal de la época (50 Hz, 0-800 ms)
#define AE_N_IN 320             // AE_N_CH * AE_N_T: entrada y salida del modelo
#define AE_FS_HZ 250            // muestreo de la ventana cruda
#define AE_DECIM 5              // diezmado por media: 250 -> 50 Hz
#define AE_PRE_SAMPLES 50       // [-200, 0) ms: línea base
#define AE_POST_SAMPLES 200     // [0, 800) ms: época
#define AE_WIN_SAMPLES 250      // AE_PRE_SAMPLES + AE_POST_SAMPLES
#define AE_MODEL_ID_LEN 16      // AE_MODEL_ID: 16 hex en minúsculas
#define AE_CALIB_MIN_SCORES 20  // mínimo de scores para calcular umbrales

typedef enum {
    AE_LEVEL_NORMAL = 0, // score <= T1
    AE_LEVEL_MILD,       // T1 < score <= T2
    AE_LEVEL_MODERATE,   // T2 < score <= T3
    AE_LEVEL_SEVERE,     // score > T3, o cualquier fallo (fail-safe)
} ae_level_t;

typedef struct {
    float score;      // MSE normalizado; +INFINITY si !valid
    ae_level_t level; // AE_LEVEL_SEVERE si !valid
    bool valid;       // false: fail-safe (ver arriba)
} ae_result_t;

// Calibración del operador = blob de NVS. Válida si: mean finitas, std finitas y > 0,
// 0 < t1 <= t2 <= t3 finitos y model_id == AE_MODEL_ID del modelo cargado (16 hex + '\0').
typedef struct {
    float mean[AE_N_CH];
    float std[AE_N_CH];
    float t1, t2, t3;
    char model_id[AE_MODEL_ID_LEN + 1];
} ae_calib_t;

// Buffers de una inferencia (~3.2 KB), del llamador. Tras ae_infer con valid = true contienen
// la cadena completa (diagnóstico); con valid = false su contenido no está definido.
typedef struct {
    float z[AE_N_IN];      // época normalizada, [canal][tiempo]
    int8_t q_in[AE_N_IN];  // entrada int8 del modelo
    int8_t q_out[AE_N_IN]; // salida int8 del modelo
    float z_hat[AE_N_IN];  // reconstrucción descuantizada
} ae_workspace_t;

// Metadatos del modelo cargado (autoencoder_weights.h + medidas de TFLM). Los rellena
// ae_runner_init durante ae_init. Antes de un ae_init correcto: model_id = arch = "" y el resto
// sin valor útil (NaN / 0).
typedef struct {
    const char *model_id;     // AE_MODEL_ID: 16 hex
    const char *arch;         // "errp_conv" | "dense"
    bool placeholder;         // true: pesos de prueba (fixture), NO un modelo entrenado
    float in_scale;           // cuantización por tensor de la entrada (= AE_IN_SCALE)
    int32_t in_zero_point;    // (= AE_IN_ZERO_POINT)
    float out_scale;          // salida (= AE_OUT_SCALE)
    int32_t out_zero_point;   // (= AE_OUT_ZERO_POINT)
    float default_mean[AE_N_CH];
    float default_std[AE_N_CH];
    float default_t[3];       // THRESHOLD_LEVEL_1..3 (p90 / p97 / p99 por defecto)
    float calib_pct[3];       // percentiles de calibración (AE_CALIB_PCT_1..3)
    uint32_t arena_bytes;     // AE_TENSOR_ARENA_BYTES
    uint32_t arena_used_bytes; // arena usada tras AllocateTensors (medida en este dispositivo)
    float int8_score_corr;    // correlación score float vs int8 del exportador (>= 0.98)
} ae_model_info_t;

// ---- ciclo de vida

// Una vez al arrancar: crea el intérprete TFLM, AllocateTensors y comprueba tipos, formas y
// cuantización de los tensores contra el header (en el runner) y la coherencia de los
// metadatos (aquí). Devuelve false (y el motor queda sin inicializar: todo fail-safe) si algo
// falla. Llamarlo otra vez reconstruye el intérprete.
bool ae_init(void);
bool ae_is_ready(void);
// Nunca NULL (ver ae_model_info_t para el estado sin inicializar).
const ae_model_info_t *ae_model_info(void);

// ---- calibración

// Calibración por defecto del modelo cargado (mean/std/umbrales del header + model_id).
// Sin inicializar: una calibración inválida a propósito (NaN, model_id ""). Pone a 0 todo el
// struct antes (blob de NVS determinista).
void ae_calib_default(ae_calib_t *c);
// true si c es válida para el modelo cargado (siempre false sin inicializar o con c == NULL).
bool ae_calib_check(const ae_calib_t *c);

// ---- inferencia (NO reentrante)

// Época 8 x 40 -> score y nivel con la calibración c. ws: buffers del llamador.
ae_result_t ae_infer(const float epoch[AE_N_CH][AE_N_T], const ae_calib_t *c, ae_workspace_t *ws);

// ---- piezas de la cadena (puras, reentrantes; sin comprobar calibración)

// win[8][250] -> epoch[8][40] (no pueden solaparse). win == NULL -> epoch lleno de NaN.
void ae_preprocess(const float win[AE_N_CH][AE_WIN_SAMPLES], float epoch[AE_N_CH][AE_N_T]);
// Siempre escribe z (si z != NULL); false si alguna muestra o alguna z no es finita, o con
// epoch / c NULL (z lleno de NaN).
bool ae_normalize(const float epoch[AE_N_CH][AE_N_T], const ae_calib_t *c, float z[AE_N_IN]);
// q = clamp(rint(z / scale) + zero_point, -128, 127); +-Inf satura; NaN -> -128 (ae_infer
// nunca cuantiza NaN: z ya es finita).
void ae_quantize(const float z[AE_N_IN], float scale, int32_t zero_point, int8_t q[AE_N_IN]);
// x = scale * (float)(q - zero_point). zero_point fuera de [-128, 127] o q == NULL -> x = NaN.
void ae_dequantize(const int8_t q[AE_N_IN], float scale, int32_t zero_point, float x[AE_N_IN]);
// (1/320) * sum (z - z_hat)^2 en bruto (puede no ser finito); NULL -> +INFINITY.
float ae_score(const float z[AE_N_IN], const float z_hat[AE_N_IN]);
// Nivel con ">" estricto. Score no finito, c == NULL o umbrales inválidos -> AE_LEVEL_SEVERE.
ae_level_t ae_level_from_score(float score, const ae_calib_t *c);

// ---- mean/std de calibración: Welford float32 sobre todas las muestras de todas las épocas,
// por canal, desviación poblacional (ddof 0). Exacto mientras n <= 2^24 muestras.
typedef struct {
    uint32_t n;          // muestras por canal acumuladas
    float mean[AE_N_CH];
    float m2[AE_N_CH];   // suma de cuadrados de desviaciones
} ae_norm_acc_t;

void ae_norm_acc_init(ae_norm_acc_t *a);
// Añade las 40 muestras de cada canal. Una muestra no finita o e == NULL envenena el
// acumulador (ae_norm_acc_finish fallará): fail-safe.
void ae_norm_acc_add(ae_norm_acc_t *a, const float e[AE_N_CH][AE_N_T]);
// c->mean / c->std = media y desviación poblacional. Falla (false, c->mean/std = NaN) sin
// muestras, con valores no finitos o con std <= 0 (canal plano). No toca umbrales ni model_id.
bool ae_norm_acc_finish(const ae_norm_acc_t *a, ae_calib_t *c);

// ---- umbrales de calibración: percentiles "linear" de numpy sobre los scores (ordena scores
// en el sitio). Usa los percentiles del modelo (AE_CALIB_PCT_1..3). Necesita el motor
// inicializado, n >= AE_CALIB_MIN_SCORES y todos los scores finitos; el resultado debe cumplir
// 0 < t1 <= t2 <= t3. Si falla devuelve false y deja c->t1..t3 = NaN (inutilizable).
bool ae_calib_thresholds_from_scores(ae_calib_t *c, float *scores, int n);
// Percentil "linear" de numpy sobre sorted[0..n) ascendente: pos = pct * (n - 1) / 100,
// lo = floor(pos), hi = min(lo + 1, n - 1), T = s[lo] + (pos - lo) * (s[hi] - s[lo]).
// lo y pos - lo salen sin pérdida cuando pct * (n - 1) es exacto en float32 (90, 97, 99, 99.5:
// T a ~1 ulp de numpy). Un pct no representable (99.9f) ya difiere del 99.9 double de Python.
// NaN si sorted == NULL, n < 1 o pct fuera de [0, 100].
float ae_percentile_sorted(const float *sorted, int n, float pct);

// ---- frontera del runner (autoencoder_tflm.cc; los tests pueden enlazar uno falso)

// Construye el intérprete y rellena info (ids, cuantización, valores por defecto, arena).
bool ae_runner_init(ae_model_info_t *info);
// Una inferencia int8: in[320] -> out[320]. false si el runner no está listo o Invoke falla.
bool ae_runner_invoke(const int8_t in[AE_N_IN], int8_t out[AE_N_IN]);

#ifdef __cplusplus
}
#endif

#endif // AUTOENCODER_ENGINE_H
