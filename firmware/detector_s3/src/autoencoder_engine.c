// Motor del ErrP-AE (detector ESP32-S3): pre/post-procesado, calibración y fail-safe en C
// puro. La inferencia int8 la hace el runner (ae_runner_*: autoencoder_tflm.cc, TFLite Micro).
// Semántica, convenciones numéricas, memoria e integración: ver autoencoder_engine.h.
//
// Reglas de este archivo:
//   - C11 puro: solo <math.h>, <string.h>, <stdbool.h>, <stdint.h>, <stddef.h> y <float.h>;
//     sin ESP-IDF, sin malloc, sin printf.
//   - Solo float32 (literales con sufijo f, rintf / sqrtf / isfinite): el S3 solo tiene FPU de
//     simple precisión y un double se emularía por software. Limpio con -Wdouble-promotion
//     -Wfloat-conversion.
//   - Único estado global mutable: s_ready y s_info, escritos solo por ae_init (documentado
//     en el .h). No incluye autoencoder_weights.h: el modelo y sus metadatos viven en el
//     runner (una sola copia en flash) y llegan aquí con ae_runner_init.
#include "autoencoder_engine.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

// El fail-safe depende de la semántica IEEE de NaN/Inf: con -ffast-math (o
// -ffinite-math-only) el compilador puede suponer que no existen, borrar los isfinite() y
// dejar pasar datos inválidos como NORMAL.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "autoencoder_engine.c no admite -ffast-math ni -ffinite-math-only: el fail-safe necesita NaN/Inf IEEE"
#endif

// Bit a bit con la referencia Python exige operaciones float evaluadas en float (no x87)
#if defined(FLT_EVAL_METHOD) && (FLT_EVAL_METHOD == 1 || FLT_EVAL_METHOD == 2)
#error "autoencoder_engine.c necesita FLT_EVAL_METHOD 0 (float evaluado en float, p.ej. SSE2 o la FPU del S3)"
#endif

_Static_assert(sizeof(float) == 4, "se asume float IEEE-754 de 32 bits");
_Static_assert(AE_N_IN == AE_N_CH * AE_N_T, "AE_N_IN debe ser AE_N_CH * AE_N_T");
_Static_assert(AE_WIN_SAMPLES == AE_PRE_SAMPLES + AE_POST_SAMPLES, "ventana = línea base + época");
_Static_assert(AE_POST_SAMPLES == AE_N_T * AE_DECIM, "la época diezmada debe cubrir AE_POST_SAMPLES");
_Static_assert(AE_CALIB_MIN_SCORES >= 1, "AE_CALIB_MIN_SCORES >= 1");

// Límite de muestras del acumulador: (float)n exacto (2^24) con margen para una época más
#define AE_NORM_ACC_MAX_N (16777216u - (uint32_t)AE_N_T)

// Metadatos "sin modelo": ae_model_info() nunca devuelve NULL y nada parece válido
#define AE_INFO_NONE_INIT                                                                              \
    {                                                                                                  \
        .model_id = "", .arch = "", .placeholder = true, .in_scale = NAN, .in_zero_point = 0,          \
        .out_scale = NAN, .out_zero_point = 0,                                                         \
        .default_mean = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN},                                      \
        .default_std = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN}, .default_t = {NAN, NAN, NAN},         \
        .calib_pct = {NAN, NAN, NAN}, .arena_bytes = 0u, .arena_used_bytes = 0u, .int8_score_corr = NAN \
    }
_Static_assert(AE_N_CH == 8, "AE_INFO_NONE_INIT inicializa 8 canales");

static const ae_model_info_t AE_INFO_NONE = AE_INFO_NONE_INIT;

// ÚNICO estado mutable del módulo: lo escribe solo ae_init (ver "NO reentrante" en el .h)
static bool s_ready = false;
static ae_model_info_t s_info = AE_INFO_NONE_INIT;

// ---------------------------------------------------------------- utilidades internas

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

static bool ae_thresholds_ok(float t1, float t2, float t3)
{
    // Escrito en positivo: con NaN toda comparación es falsa y el resultado es false
    return isfinite(t1) && isfinite(t2) && isfinite(t3) && t1 > 0.0f && t1 <= t2 && t2 <= t3;
}

// mean finitas, std finitas > 0 y umbrales válidos (sin model_id)
static bool ae_calib_values_ok(const float mean[AE_N_CH], const float std[AE_N_CH], float t1, float t2, float t3)
{
    for (int ch = 0; ch < AE_N_CH; ch++) {
        if (!isfinite(mean[ch]) || !isfinite(std[ch]) || !(std[ch] > 0.0f)) {
            return false;
        }
    }
    return ae_thresholds_ok(t1, t2, t3);
}

// Exactamente AE_MODEL_ID_LEN caracteres hex en minúsculas seguidos de '\0'
static bool ae_model_id_ok(const char *id)
{
    if (id == NULL) {
        return false;
    }
    for (int i = 0; i < AE_MODEL_ID_LEN; i++) {
        const char ch = id[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
            return false; // también corta en un '\0' prematuro
        }
    }
    return id[AE_MODEL_ID_LEN] == '\0';
}

// Coherencia de los metadatos que entrega el runner (defensa en profundidad: el runner ya
// comparó los tensores con el header)
static bool ae_info_ok(const ae_model_info_t *info)
{
    if (!ae_model_id_ok(info->model_id) || info->arch == NULL || info->arch[0] == '\0') {
        return false;
    }
    if (!isfinite(info->in_scale) || !(info->in_scale > 0.0f) || !isfinite(info->out_scale) ||
        !(info->out_scale > 0.0f)) {
        return false;
    }
    if (info->in_zero_point < -128 || info->in_zero_point > 127 || info->out_zero_point < -128 ||
        info->out_zero_point > 127) {
        return false;
    }
    if (!ae_calib_values_ok(info->default_mean, info->default_std, info->default_t[0], info->default_t[1],
                            info->default_t[2])) {
        return false;
    }
    const float *p = info->calib_pct;
    if (!(isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]) && p[0] > 0.0f && p[0] <= p[1] && p[1] <= p[2] &&
          p[2] <= 100.0f)) {
        return false;
    }
    return info->arena_bytes > 0u && info->arena_used_bytes <= info->arena_bytes;
}

// Heapsort ascendente en el sitio: O(n log n), sin recursión ni memoria extra. Solo con
// valores finitos (orden total).
static void ae_sift_down(float *v, int root, int end)
{
    while (2 * root + 1 < end) {
        int child = 2 * root + 1;
        if (child + 1 < end && v[child] < v[child + 1]) {
            child++;
        }
        if (!(v[root] < v[child])) {
            return;
        }
        const float tmp = v[root];
        v[root] = v[child];
        v[child] = tmp;
        root = child;
    }
}

static void ae_sort_floats(float *v, int n)
{
    for (int i = n / 2 - 1; i >= 0; i--) {
        ae_sift_down(v, i, n);
    }
    for (int end = n - 1; end > 0; end--) {
        const float tmp = v[0];
        v[0] = v[end];
        v[end] = tmp;
        ae_sift_down(v, 0, end);
    }
}

// ---------------------------------------------------------------- ciclo de vida

bool ae_init(void)
{
    ae_model_info_t info = AE_INFO_NONE_INIT;

    // Mientras se inicializa (y si falla) el motor queda sin modelo: todo fail-safe
    s_ready = false;
    s_info = AE_INFO_NONE;
    if (!ae_runner_init(&info) || !ae_info_ok(&info)) {
        return false;
    }
    s_info = info;
    s_ready = true;
    return true;
}

bool ae_is_ready(void)
{
    return s_ready;
}

const ae_model_info_t *ae_model_info(void)
{
    return &s_info;
}

// ---------------------------------------------------------------- calibración

void ae_calib_default(ae_calib_t *c)
{
    if (c == NULL) {
        return;
    }
    memset(c, 0, sizeof(*c)); // relleno y model_id a cero: blob de NVS determinista
    if (!s_ready) {
        // Inválida a propósito (model_id "" y NaN): ae_calib_check la rechaza
        ae_fill_nan(c->mean, AE_N_CH);
        ae_fill_nan(c->std, AE_N_CH);
        c->t1 = NAN;
        c->t2 = NAN;
        c->t3 = NAN;
        return;
    }
    memcpy(c->mean, s_info.default_mean, sizeof(c->mean));
    memcpy(c->std, s_info.default_std, sizeof(c->std));
    c->t1 = s_info.default_t[0];
    c->t2 = s_info.default_t[1];
    c->t3 = s_info.default_t[2];
    memcpy(c->model_id, s_info.model_id, AE_MODEL_ID_LEN); // s_info.model_id: 16 hex validados
    c->model_id[AE_MODEL_ID_LEN] = '\0';
}

bool ae_calib_check(const ae_calib_t *c)
{
    if (!s_ready || c == NULL) {
        return false;
    }
    if (!ae_calib_values_ok(c->mean, c->std, c->t1, c->t2, c->t3)) {
        return false;
    }
    // Calibración de otro modelo (o blob corrupto): rechazar. memcmp de los 16 caracteres y
    // terminador exacto; s_info.model_id ya es hex válido, así que esto también valida c.
    return memcmp(c->model_id, s_info.model_id, AE_MODEL_ID_LEN) == 0 && c->model_id[AE_MODEL_ID_LEN] == '\0';
}

// ---------------------------------------------------------------- inferencia

ae_result_t ae_infer(const float epoch[AE_N_CH][AE_N_T], const ae_calib_t *c, ae_workspace_t *ws)
{
    if (!s_ready || epoch == NULL || ws == NULL || !ae_calib_check(c)) {
        return ae_invalid_result();
    }
    // Época no finita (o z que desborda): la red no se ejecuta
    if (!ae_normalize(epoch, c, ws->z)) {
        return ae_invalid_result();
    }
    ae_quantize(ws->z, s_info.in_scale, s_info.in_zero_point, ws->q_in);
    if (!ae_runner_invoke(ws->q_in, ws->q_out)) {
        return ae_invalid_result();
    }
    ae_dequantize(ws->q_out, s_info.out_scale, s_info.out_zero_point, ws->z_hat);
    const float score = ae_score(ws->z, ws->z_hat);
    if (!isfinite(score)) {
        return ae_invalid_result(); // desbordamiento con z enormes pero finitas
    }
    const ae_result_t r = {.score = score, .level = ae_level_from_score(score, c), .valid = true};
    return r;
}

// ---------------------------------------------------------------- piezas de la cadena

void ae_preprocess(const float win[AE_N_CH][AE_WIN_SAMPLES], float epoch[AE_N_CH][AE_N_T])
{
    if (epoch == NULL) {
        return;
    }
    if (win == NULL) {
        for (int ch = 0; ch < AE_N_CH; ch++) {
            ae_fill_nan(epoch[ch], AE_N_T);
        }
        return;
    }
    for (int ch = 0; ch < AE_N_CH; ch++) {
        const float *w = win[ch];

        // Línea base: media de [-200, 0) ms = muestras 0..49 (suma en orden de índice)
        float sum = 0.0f;
        for (int j = 0; j < AE_PRE_SAMPLES; j++) {
            sum += w[j];
        }
        const float base = sum / (float)AE_PRE_SAMPLES;

        // Diezmado x5 por media de [0, 800) ms: muestra k = media de w[50+5k .. 54+5k]
        for (int k = 0; k < AE_N_T; k++) {
            const float *seg = &w[AE_PRE_SAMPLES + AE_DECIM * k];
            float s = 0.0f;
            for (int i = 0; i < AE_DECIM; i++) {
                s += seg[i];
            }
            epoch[ch][k] = s / (float)AE_DECIM - base;
        }
    }
}

bool ae_normalize(const float epoch[AE_N_CH][AE_N_T], const ae_calib_t *c, float z[AE_N_IN])
{
    if (z == NULL) {
        return false;
    }
    if (epoch == NULL || c == NULL) {
        ae_fill_nan(z, AE_N_IN);
        return false;
    }
    bool ok = true;
    for (int ch = 0; ch < AE_N_CH; ch++) {
        const float m = c->mean[ch];
        const float s = c->std[ch];
        for (int t = 0; t < AE_N_T; t++) {
            const float x = epoch[ch][t];
            const float v = (x - m) / s; // división float32 (no multiplicar por 1/std: otro redondeo)
            z[ch * AE_N_T + t] = v;
            if (!isfinite(x) || !isfinite(v)) {
                ok = false;
            }
        }
    }
    return ok;
}

void ae_quantize(const float z[AE_N_IN], float scale, int32_t zero_point, int8_t q[AE_N_IN])
{
    if (z == NULL || q == NULL) {
        return;
    }
    const float zp = (float)zero_point;
    for (int i = 0; i < AE_N_IN; i++) {
        // rintf: redondeo a par (modo FP por defecto), como np.rint. La suma con zp es exacta
        // en el rango útil; el clamp se hace en float para no convertir a entero valores
        // fuera de rango (comportamiento indefinido en C).
        const float v = rintf(z[i] / scale) + zp;
        int8_t out;
        if (v >= 127.0f) {
            out = 127;
        } else if (v > -128.0f) {
            out = (int8_t)(int32_t)v; // entero exacto en (-128, 127)
        } else {
            out = -128; // también NaN (todas las comparaciones fallan)
        }
        q[i] = out;
    }
}

void ae_dequantize(const int8_t q[AE_N_IN], float scale, int32_t zero_point, float x[AE_N_IN])
{
    if (x == NULL) {
        return;
    }
    if (q == NULL || zero_point < -128 || zero_point > 127) {
        ae_fill_nan(x, AE_N_IN); // sin zero_point int8 la resta podría desbordar: fail-safe
        return;
    }
    for (int i = 0; i < AE_N_IN; i++) {
        x[i] = scale * (float)((int32_t)q[i] - zero_point);
    }
}

float ae_score(const float z[AE_N_IN], const float z_hat[AE_N_IN])
{
    if (z == NULL || z_hat == NULL) {
        return INFINITY;
    }
    // Un solo acumulador en orden de índice (mismo orden que la referencia Python)
    float acc = 0.0f;
    for (int i = 0; i < AE_N_IN; i++) {
        const float d = z[i] - z_hat[i];
        acc += d * d;
    }
    return acc / (float)AE_N_IN;
}

ae_level_t ae_level_from_score(float score, const ae_calib_t *c)
{
    // isfinite primero: con NaN todas las comparaciones son falsas y caería en NORMAL
    if (c == NULL || !isfinite(score) || !ae_thresholds_ok(c->t1, c->t2, c->t3)) {
        return AE_LEVEL_SEVERE;
    }
    if (score > c->t3) {
        return AE_LEVEL_SEVERE;
    }
    if (score > c->t2) {
        return AE_LEVEL_MODERATE;
    }
    if (score > c->t1) {
        return AE_LEVEL_MILD;
    }
    return AE_LEVEL_NORMAL;
}

// ---------------------------------------------------------------- mean/std (Welford)

void ae_norm_acc_init(ae_norm_acc_t *a)
{
    if (a == NULL) {
        return;
    }
    memset(a, 0, sizeof(*a));
}

void ae_norm_acc_add(ae_norm_acc_t *a, const float e[AE_N_CH][AE_N_T])
{
    if (a == NULL) {
        return;
    }
    if (e == NULL || a->n > AE_NORM_ACC_MAX_N) {
        ae_fill_nan(a->m2, AE_N_CH); // envenenado: ae_norm_acc_finish fallará
        return;
    }
    // Muestra a muestra en orden temporal; todos los canales comparten n
    for (int t = 0; t < AE_N_T; t++) {
        a->n++;
        const float n = (float)a->n; // exacto: n <= 2^24
        for (int ch = 0; ch < AE_N_CH; ch++) {
            const float x = e[ch][t];
            const float delta = x - a->mean[ch];
            a->mean[ch] += delta / n;
            a->m2[ch] += delta * (x - a->mean[ch]);
        }
    }
}

bool ae_norm_acc_finish(const ae_norm_acc_t *a, ae_calib_t *c)
{
    if (c == NULL) {
        return false;
    }
    float mean[AE_N_CH];
    float std[AE_N_CH];
    bool ok = a != NULL && a->n > 0u;
    for (int ch = 0; ok && ch < AE_N_CH; ch++) {
        mean[ch] = a->mean[ch];
        std[ch] = sqrtf(a->m2[ch] / (float)a->n); // poblacional (ddof 0): m2 / n
        ok = isfinite(mean[ch]) && isfinite(std[ch]) && std[ch] > 0.0f;
    }
    if (!ok) {
        ae_fill_nan(c->mean, AE_N_CH); // fail-safe: una calibración a medias no se puede usar
        ae_fill_nan(c->std, AE_N_CH);
        return false;
    }
    memcpy(c->mean, mean, sizeof(mean));
    memcpy(c->std, std, sizeof(std));
    return true;
}

// ---------------------------------------------------------------- umbrales (percentiles)

float ae_percentile_sorted(const float *sorted, int n, float pct)
{
    if (sorted == NULL || n < 1 || !(pct >= 0.0f && pct <= 100.0f)) {
        return NAN;
    }
    // pos = p100 / 100 = lo + frac sin perder precisión: p100 = pct * (n - 1) es exacto para
    // percentiles con pocos decimales, lo y el resto p100 - 100 lo son exactos (enteros < 2^24 y
    // Sterbenz) y solo frac = resto / 100 redondea, con error relativo a frac. Calcular pos en
    // float32 y restarle lo perdería ~ulp(pos) en frac (4e-4 relativo en T con colas pesadas).
    const float p100 = pct * (float)(n - 1);
    int lo = (int)(p100 / 100.0f); // >= 0: truncar = floor (salvo el redondeo de la división)
    if (lo > 0 && (float)lo * 100.0f > p100) {
        lo--;
    } else if ((float)(lo + 1) * 100.0f <= p100) {
        lo++;
    }
    if (lo > n - 1) {
        lo = n - 1;
    }
    const int hi = (lo + 1 < n) ? lo + 1 : n - 1;
    const float frac = (p100 - (float)lo * 100.0f) / 100.0f;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

bool ae_calib_thresholds_from_scores(ae_calib_t *c, float *scores, int n)
{
    if (c == NULL) {
        return false;
    }
    bool ok = s_ready && scores != NULL && n >= AE_CALIB_MIN_SCORES;
    for (int i = 0; ok && i < n; i++) {
        ok = isfinite(scores[i]);
    }
    float t[3] = {NAN, NAN, NAN};
    if (ok) {
        ae_sort_floats(scores, n);
        for (int k = 0; k < 3; k++) {
            t[k] = ae_percentile_sorted(scores, n, s_info.calib_pct[k]);
        }
        ok = ae_thresholds_ok(t[0], t[1], t[2]);
    }
    if (!ok) {
        c->t1 = NAN; // fail-safe: sin umbrales válidos la calibración no pasa ae_calib_check
        c->t2 = NAN;
        c->t3 = NAN;
        return false;
    }
    c->t1 = t[0];
    c->t2 = t[1];
    c->t3 = t[2];
    return true;
}
