// Test del motor C del autoencoder (src/autoencoder_engine.c): vectores dorados de
// PyTorch generados por ml/c_exporter/export_to_c.py (autoencoder_test_vectors.h, en
// este mismo directorio) más casos unitarios: activaciones, capa densa, umbrales,
// fail-safe, determinismo, constantes del modelo y un benchmark.
//
// Harness propio (sin Unity) con salida compatible con Unity / PlatformIO:
//   test_c_engine.c:<línea>:<test>:PASS
//   test_c_engine.c:<línea>:<test>:FAIL: <mensaje>
//   -----------------------
//   <N> Tests <M> Failures 0 Ignored
//   OK | FAIL
// Como un TEST_ASSERT de Unity, el primer CHECK fallido de un test imprime su línea FAIL;
// los siguientes fallos (hasta MAX_DETAIL_LINES) salen debajo, indentados, como detalle.
//
// Build en el PC, desde la raíz del repo (una sola línea):
//   $CC -std=c11 -O2 -Wall -Wextra -Wpedantic -Wdouble-promotion -Wfloat-conversion -Wshadow -Werror -I firmware/detector_s3/include firmware/detector_s3/src/autoencoder_engine.c firmware/detector_s3/test/test_c_engine.c -lm -o test_c_engine
//   ./test_c_engine            (código de salida != 0 si falla algún test)
// con CC = gcc en Linux, o "python -m ziglang cc" en Windows. zig respeta el nombre de -o
// tal cual (sin .exe): ./test_c_engine funciona en Git Bash; para PowerShell/cmd usar
// -o test_c_engine.exe.
//
// En el ESP32-S3 (ESP-IDF define ESP_PLATFORM): app_main() corre la misma batería en
// una tarea con 16 KB de pila y el benchmark usa esp_timer_get_time() (el componente
// necesita esp_timer en REQUIRES/PRIV_REQUIRES).
//
// Si falla test_model_id, pesos y vectores son de modelos distintos: regenerar los dos
// a la vez con ml/c_exporter/export_to_c.py.
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "autoencoder_engine.h"
#include "autoencoder_test_vectors.h"

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#define BENCH_ITERS 2000
#else
#include <time.h>
#define BENCH_ITERS 200000
#endif

_Static_assert(AE_TV_DIM == AE_N_FEATURES, "AE_TV_DIM != AE_N_FEATURES: regenerar los vectores con el exportador");
_Static_assert(AE_TV_COUNT >= 1, "autoencoder_test_vectors.h sin vectores");

// Tolerancia frente a PyTorch: |a - b| <= TOL_ABS + TOL_REL * |b|. El motor suma en float32,
// en orden secuencial con un solo acumulador (y en el S3 fusiona multiplicación y suma en
// madd.s, FMA); la referencia de PyTorch suma en otro orden (BLAS, FMA vectorial) o incluso
// en float64. Eso mueve el resultado ~1e-7 relativo; 1e-4 deja margen sin tapar bugs reales
// (un peso traspuesto, un sesgo olvidado o una activación cambiada dan errores >> 1e-3).
#define TOL_ABS 1e-4f
#define TOL_REL 1e-4f
// Activaciones sueltas: solo cambia la libm (expf/expm1f/tanhf/erff, ~1-2 ulp)
#define TOL_ACT_ABS 1e-6f
#define TOL_ACT_REL 1e-5f

#define MAX_DETAIL_LINES 8 // fallos impresos por test (el primero es la línea FAIL)

// ---------------------------------------------------------------- harness

static const char *g_file;      // nombre base de __FILE__ (sin ruta: "C:\..." rompería el formato)
static const char *g_test_name; // test en curso
static int g_fails;             // CHECK fallidos en el test en curso

#if defined(__GNUC__)
static void check_failed(int line, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#endif

static void check_failed(int line, const char *fmt, ...)
{
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    g_fails++;
    if (g_fails == 1) {
        printf("%s:%d:%s:FAIL: %s\n", g_file, line, g_test_name, msg);
    } else if (g_fails <= MAX_DETAIL_LINES) {
        printf("  %s:%d: %s\n", g_file, line, msg);
    }
}

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            check_failed(__LINE__, __VA_ARGS__); \
        }                                        \
    } while (0)

static bool near_tol(float actual, float expected, float tol_abs, float tol_rel)
{
    return fabsf(actual - expected) <= tol_abs + tol_rel * fabsf(expected); // NaN -> false
}

// CHECK_NEAR(obtenido, esperado, "etiqueta printf", args...) con la tolerancia de los vectores dorados
#define CHECK_NEAR(actual, expected, label, ...)                                                     \
    CHECK(near_tol((actual), (expected), TOL_ABS, TOL_REL), label ": obtenido %.9g, esperado %.9g", \
          __VA_ARGS__, (double)(actual), (double)(expected))

static bool same_bits(float a, float b)
{
    uint32_t ua;
    uint32_t ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

static bool same_result(ae_result_t a, ae_result_t b)
{
    return same_bits(a.score, b.score) && a.level == b.level && a.valid == b.valid;
}

static bool is_pos_inf(float v)
{
    return isinf(v) && v > 0.0f;
}

static bool all_nan(const float *v, int n)
{
    for (int i = 0; i < n; i++) {
        if (!isnan(v[i])) {
            return false;
        }
    }
    return true;
}

// Máximo que no se traga los NaN (fmaxf los ignoraría)
static float max_err(float acc, float err)
{
    return (err > acc || isnan(err)) ? err : acc;
}

static const char *base_name(const char *path)
{
    const char *base = path;
    for (const char *p = path; *p != '\0'; p++) {
        if (*p == '/' || *p == '\\') {
            base = p + 1;
        }
    }
    return base;
}

static int64_t now_us(void)
{
#ifdef ESP_PLATFORM
    return esp_timer_get_time();
#else
    return (int64_t)clock() * 1000000 / CLOCKS_PER_SEC;
#endif
}

// Resultado inválido según el contrato: valid = false, score = +INFINITY, SEVERE
static void check_invalid_result(int line, ae_result_t r, const char *what)
{
    if (r.valid || !is_pos_inf(r.score) || r.level != AE_LEVEL_SEVERE) {
        check_failed(line, "%s: esperado valid=0 score=+inf level=3, obtenido valid=%d score=%.9g level=%d", what,
                     (int)r.valid, (double)r.score, (int)r.level);
    }
}

// ---------------------------------------------------------------- (1) identidad del modelo

static void test_model_id(void)
{
    const char *id = ae_model_id();
    CHECK(strcmp(id, AE_TV_MODEL_ID) == 0,
          "AE_MODEL_ID del motor (%s) != AE_TV_MODEL_ID de los vectores (%s): pesos y vectores son de modelos "
          "distintos, regenerar ambos con ml/c_exporter/export_to_c.py",
          id, AE_TV_MODEL_ID);

    bool hex_ok = strlen(id) == 16;
    for (size_t i = 0; hex_ok && i < 16; i++) {
        hex_ok = (id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f');
    }
    CHECK(hex_ok, "AE_MODEL_ID \"%s\" no tiene 16 caracteres hex en minusculas", id);
}

// ---------------------------------------------------------------- (2) vectores dorados de PyTorch

static void test_golden_vectors(void)
{
    float max_err_z = 0.0f;
    float max_err_zhat = 0.0f;
    float max_err_score = 0.0f;
    int level_checked = 0;

    for (int k = 0; k < AE_TV_COUNT; k++) {
        // Piezas sueltas: normalización y reconstrucción
        float z[AE_N_FEATURES];
        float z_hat[AE_N_FEATURES];
        const bool norm_ok = ae_normalize(AE_TV_X[k], z);
        CHECK(norm_ok, "vector %d: ae_normalize rechazo una entrada valida", k);
        ae_forward(z, z_hat);
        for (int i = 0; i < AE_N_FEATURES; i++) {
            max_err_z = max_err(max_err_z, fabsf(z[i] - AE_TV_Z[k][i]));
            max_err_zhat = max_err(max_err_zhat, fabsf(z_hat[i] - AE_TV_ZHAT[k][i]));
            CHECK_NEAR(z[i], AE_TV_Z[k][i], "vector %d, z[%d]", k, i);
            CHECK_NEAR(z_hat[i], AE_TV_ZHAT[k][i], "vector %d, z_hat[%d] (PyTorch)", k, i);
        }

        // Inferencia completa: score, valid y nivel
        const ae_result_t r = ae_infer(AE_TV_X[k]);
        max_err_score = max_err(max_err_score, fabsf(r.score - AE_TV_SCORE[k]));
        CHECK(r.valid, "vector %d: ae_infer devolvio valid=false con una entrada valida", k);
        CHECK_NEAR(r.score, AE_TV_SCORE[k], "vector %d, score", k);
        if (!AE_TV_NEAR_THRESHOLD[k]) {
            // Cerca de un umbral el redondeo puede cambiar el nivel: solo se exige lejos de ellos
            level_checked++;
            CHECK((int)r.level == AE_TV_LEVEL[k], "vector %d: nivel %d, esperado %d (score %.9g, esperado %.9g)", k,
                  (int)r.level, AE_TV_LEVEL[k], (double)r.score, (double)AE_TV_SCORE[k]);
        }

        // ae_infer_ex devuelve exactamente lo mismo que las piezas sueltas (también con un solo buffer)
        float z_ex[AE_N_FEATURES];
        float z_hat_ex[AE_N_FEATURES];
        const ae_result_t r_ex = ae_infer_ex(AE_TV_X[k], z_ex, z_hat_ex);
        CHECK(memcmp(z_ex, z, sizeof(z)) == 0, "vector %d: z de ae_infer_ex != ae_normalize", k);
        CHECK(memcmp(z_hat_ex, z_hat, sizeof(z_hat)) == 0, "vector %d: z_hat de ae_infer_ex != ae_forward", k);
        CHECK(same_result(r_ex, r), "vector %d: ae_infer_ex != ae_infer", k);
        float z_hat_only[AE_N_FEATURES];
        const ae_result_t r_hat_only = ae_infer_ex(AE_TV_X[k], NULL, z_hat_only);
        CHECK(same_result(r_hat_only, r) && memcmp(z_hat_only, z_hat, sizeof(z_hat)) == 0,
              "vector %d: ae_infer_ex(x, NULL, z_hat) != ae_forward", k);
        CHECK(same_bits(r.score, ae_score(z, z_hat)), "vector %d: score de ae_infer (%.9g) != ae_score (%.9g)", k,
              (double)r.score, (double)ae_score(z, z_hat));
    }
    printf("  info: %d vectores, error max abs: z=%.3g z_hat=%.3g score=%.3g; nivel exacto en %d (los demas cerca "
           "de un umbral)\n",
           AE_TV_COUNT, (double)max_err_z, (double)max_err_zhat, (double)max_err_score, level_checked);
}

// Cada banda de nivel no vacía tiene >= 2 vectores: si no, el test dorado no ejercita ese nivel
static void test_golden_coverage(void)
{
    int count[4] = {0, 0, 0, 0};
    for (int k = 0; k < AE_TV_COUNT; k++) {
        const int lv = AE_TV_LEVEL[k];
        CHECK(lv >= 0 && lv <= 3, "vector %d: AE_TV_LEVEL = %d fuera de 0..3", k, lv);
        if (lv >= 0 && lv <= 3) {
            count[lv]++;
        }
    }
    for (int lv = 0; lv <= 3; lv++) {
        // Banda [T_lv, T_lv+1), con T_0 = 0 y ae_threshold(4) = +inf
        const bool band_nonempty = ae_threshold((ae_level_t)lv) < ae_threshold((ae_level_t)(lv + 1));
        CHECK(!band_nonempty || count[lv] >= 2,
              "solo %d vectores de nivel %d (minimo 2 por banda): regenerar con el exportador y revisar sus AVISOS "
              "(umbrales o mean/std incoherentes con el modelo)",
              count[lv], lv);
    }
    printf("  info: vectores por nivel: N0=%d N1=%d N2=%d N3=%d\n", count[0], count[1], count[2], count[3]);
}

// ---------------------------------------------------------------- (3) activaciones vs PyTorch

#define ACT_N 15
static const float ACT_X[ACT_N] = {-1e20f, -100.0f, -10.0f, -3.0f, -1.0f, -0.5f, -1e-3f, 0.0f,
                                   1e-3f,  0.5f,    1.0f,   3.0f,  10.0f, 100.0f, 1e20f};

typedef struct {
    int act;
    float param;
    const char *name;
    float expected[ACT_N];
} act_case_t;

// Referencias calculadas con PyTorch 2.14 en float32 sobre ACT_X:
//   F.relu, F.leaky_relu(x, s), F.elu(x, a), torch.tanh, torch.sigmoid, F.silu,
//   F.gelu(x) y F.gelu(x, approximate="tanh"); impresas con "%.9g".
static const act_case_t ACT_CASES[] = {
    {AE_ACT_IDENTITY, 0.0f, "identity",
     {-1.00000002e+20f, -100.0f, -10.0f, -3.0f, -1.0f, -0.5f, -0.00100000005f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_RELU, 0.0f, "relu",
     {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_LEAKY_RELU, 0.01f, "leaky_relu(0.01)",
     {-9.99999984e+17f, -1.0f, -0.099999994f, -0.0299999993f, -0.00999999978f, -0.00499999989f, -1.00000007e-05f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_LEAKY_RELU, 0.2f, "leaky_relu(0.2)",
     {-2e+19f, -20.0f, -2.0f, -0.600000024f, -0.200000003f, -0.100000001f, -0.000200000009f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_ELU, 1.0f, "elu(1.0)",
     {-1.0f, -1.0f, -0.999954581f, -0.950212955f, -0.63212055f, -0.393469334f, -0.00099950016f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_ELU, 0.5f, "elu(0.5)",
     {-0.5f, -0.5f, -0.499977291f, -0.475106478f, -0.316060275f, -0.196734667f, -0.00049975008f, 0.0f,
      0.00100000005f, 0.5f, 1.0f, 3.0f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_TANH, 0.0f, "tanh",
     {-1.0f, -1.0f, -1.0f, -0.995054781f, -0.761594176f, -0.462117165f, -0.000999999698f, 0.0f,
      0.000999999698f, 0.462117165f, 0.761594176f, 0.995054781f, 1.0f, 1.0f, 1.0f}},
    {AE_ACT_SIGMOID, 0.0f, "sigmoid",
     {0.0f, 0.0f, 4.53978719e-05f, 0.0474258736f, 0.268941432f, 0.377540678f, 0.499750018f, 0.5f,
      0.500249982f, 0.622459352f, 0.731058598f, 0.952574134f, 0.999954581f, 1.0f, 1.0f}},
    {AE_ACT_SILU, 0.0f, "silu",
     {-0.0f, -0.0f, -0.000453978719f, -0.142277613f, -0.268941432f, -0.188770339f, -0.000499750022f, 0.0f,
      0.000500250026f, 0.311229676f, 0.731058598f, 2.85772252f, 9.99954605f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_GELU, 0.0f, "gelu",
     {0.0f, 0.0f, 0.0f, -0.00404986739f, -0.158655286f, -0.154268786f, -0.000499601127f, 0.0f,
      0.000500398921f, 0.345731199f, 0.841344714f, 2.99595022f, 10.0f, 100.0f, 1.00000002e+20f}},
    {AE_ACT_GELU_TANH, 0.0f, "gelu_tanh",
     {-0.0f, -0.0f, -0.0f, -0.00363743305f, -0.158807993f, -0.154285997f, -0.000499601068f, 0.0f,
      0.000500398979f, 0.345714003f, 0.841192007f, 2.99636269f, 10.0f, 100.0f, 1.00000002e+20f}},
};

static void test_activations(void)
{
    const int n_cases = (int)(sizeof(ACT_CASES) / sizeof(ACT_CASES[0]));
    for (int c = 0; c < n_cases; c++) {
        const act_case_t *tc = &ACT_CASES[c];
        float v[ACT_N];
        memcpy(v, ACT_X, sizeof(v));
        ae_activate(tc->act, tc->param, v, ACT_N);
        for (int i = 0; i < ACT_N; i++) {
            CHECK(near_tol(v[i], tc->expected[i], TOL_ACT_ABS, TOL_ACT_REL),
                  "%s(%.9g): obtenido %.9g, esperado %.9g (PyTorch)", tc->name, (double)ACT_X[i], (double)v[i],
                  (double)tc->expected[i]);
        }
        // NaN se propaga en todas (como en PyTorch): acaba en resultado inválido, no en NORMAL
        float nan_in[1] = {NAN};
        ae_activate(tc->act, tc->param, nan_in, 1);
        CHECK(isnan(nan_in[0]), "%s(NaN) = %.9g: NaN debe propagarse", tc->name, (double)nan_in[0]);
    }

    // Código desconocido -> todo NAN (fail-safe)
    const int unknown[3] = {-1, AE_ACT_GELU_TANH + 1, 1000};
    for (int u = 0; u < 3; u++) {
        float v[3] = {1.0f, -2.0f, 0.0f};
        ae_activate(unknown[u], 0.5f, v, 3);
        CHECK(all_nan(v, 3), "codigo de activacion desconocido %d: la salida debe ser NAN", unknown[u]);
    }

    // n = 0 no toca el buffer
    float guard = 7.0f;
    ae_activate(AE_ACT_GELU, 0.0f, &guard, 0);
    ae_activate(1000, 0.0f, &guard, 0);
    CHECK(guard == 7.0f, "ae_activate con n = 0 modifico el buffer (%.9g)", (double)guard);
}

// ---------------------------------------------------------------- (4) capa densa

static void test_dense(void)
{
    // W no cuadrada (una W leída traspuesta da otro resultado) y sesgo no nulo
    static const float w[6] = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    static const float b2[2] = {0.5f, -1.0f};
    static const float b3[3] = {0.0f, 1.0f, -1.0f};
    static const float in[3] = {1.0f, -1.0f, 2.0f};

    // Como 2x3: y0 = 0.5 + 1 - 2 + 6 = 5.5 ; y1 = -1 + 4 - 5 + 12 = 10 (exactos en float)
    float y23[2] = {NAN, NAN};
    ae_dense(w, b2, in, y23, 3, 2);
    CHECK(y23[0] == 5.5f && y23[1] == 10.0f, "ae_dense 2x3: y = [%.9g, %.9g], esperado [5.5, 10]",
          (double)y23[0], (double)y23[1]);

    // Los mismos datos como 3x2: y0 = 0 + 1 - 2 = -1 ; y1 = 1 + 3 - 4 = 0 ; y2 = -1 + 5 - 6 = -2
    float y32[3] = {NAN, NAN, NAN};
    ae_dense(w, b3, in, y32, 2, 3);
    CHECK(y32[0] == -1.0f && y32[1] == 0.0f && y32[2] == -2.0f,
          "ae_dense 3x2: y = [%.9g, %.9g, %.9g], esperado [-1, 0, -2]", (double)y32[0], (double)y32[1],
          (double)y32[2]);

    // n_in = 0: solo el sesgo
    float y_bias[2] = {NAN, NAN};
    ae_dense(w, b2, in, y_bias, 0, 2);
    CHECK(y_bias[0] == 0.5f && y_bias[1] == -1.0f, "ae_dense con n_in = 0: y = [%.9g, %.9g], esperado el sesgo",
          (double)y_bias[0], (double)y_bias[1]);
}

// ---------------------------------------------------------------- (5) fronteras de nivel

// Transcripción literal del contrato, referencia para el barrido
static ae_level_t contract_level(float s, float t1, float t2, float t3)
{
    if (!isfinite(s)) {
        return AE_LEVEL_SEVERE;
    }
    if (s >= t3) {
        return AE_LEVEL_SEVERE;
    }
    if (s >= t2) {
        return AE_LEVEL_MODERATE;
    }
    if (s >= t1) {
        return AE_LEVEL_MILD;
    }
    return AE_LEVEL_NORMAL;
}

#define CHECK_LEVEL(score, expected)                                                               \
    CHECK(ae_level_from_score(score) == (expected), "ae_level_from_score(%.9g) = %d, esperado %d", \
          (double)(score), (int)ae_level_from_score(score), (int)(expected))

static void test_level_boundaries(void)
{
    const float t1 = ae_threshold(AE_LEVEL_MILD);
    const float t2 = ae_threshold(AE_LEVEL_MODERATE);
    const float t3 = ae_threshold(AE_LEVEL_SEVERE);

    // Un score igual al umbral escala (>=); un ulp por debajo se queda en el nivel anterior
    CHECK_LEVEL(t3, AE_LEVEL_SEVERE);
    CHECK_LEVEL(nextafterf(t1, 0.0f), AE_LEVEL_NORMAL);
    if (t1 < t2) {
        CHECK_LEVEL(t1, AE_LEVEL_MILD);
        CHECK_LEVEL(nextafterf(t2, 0.0f), AE_LEVEL_MILD);
    }
    if (t2 < t3) {
        CHECK_LEVEL(t2, AE_LEVEL_MODERATE);
        CHECK_LEVEL(nextafterf(t3, 0.0f), AE_LEVEL_MODERATE);
    }

    // Extremos y no finitos (fail-safe: todo lo no finito es SEVERE)
    CHECK_LEVEL(0.0f, AE_LEVEL_NORMAL);
    CHECK_LEVEL(-0.0f, AE_LEVEL_NORMAL);
    CHECK_LEVEL(FLT_MAX, AE_LEVEL_SEVERE);
    CHECK_LEVEL(NAN, AE_LEVEL_SEVERE);
    CHECK_LEVEL(INFINITY, AE_LEVEL_SEVERE);
    CHECK_LEVEL(-INFINITY, AE_LEVEL_SEVERE);

    // Barrido: +-4 ulp alrededor de cada umbral y escala logarítmica de 1e-3*T1 a 1e5*T1
    const float ts[3] = {t1, t2, t3};
    for (int k = 0; k < 3; k++) {
        float s = ts[k];
        for (int j = 0; j < 4; j++) {
            s = nextafterf(s, 0.0f);
        }
        for (int j = 0; j < 9; j++) {
            CHECK_LEVEL(s, contract_level(s, t1, t2, t3));
            s = nextafterf(s, INFINITY);
        }
    }
    for (int j = 0; j <= 400; j++) {
        const float s = t1 * powf(10.0f, -3.0f + 0.02f * (float)j);
        CHECK_LEVEL(s, contract_level(s, t1, t2, t3));
    }
}

// ---------------------------------------------------------------- (6) fail-safe

static void test_fail_safe(void)
{
    const float bad[3] = {NAN, INFINITY, -INFINITY};
    const char *bad_name[3] = {"NaN", "+Inf", "-Inf"};
    const int pos[3] = {0, AE_N_FEATURES / 2, AE_N_FEATURES - 1};
    char what[96];

    // NaN / +Inf / -Inf en la primera, una intermedia y la última feature
    for (int b = 0; b < 3; b++) {
        for (int p = 0; p < 3; p++) {
            float x[AE_N_FEATURES];
            memcpy(x, AE_TV_X[0], sizeof(x));
            x[pos[p]] = bad[b];
            snprintf(what, sizeof(what), "%s en feature %d", bad_name[b], pos[p]);

            float z[AE_N_FEATURES];
            CHECK(!ae_normalize(x, z), "%s: ae_normalize devolvio true", what);

            check_invalid_result(__LINE__, ae_infer(x), what);

            // Con entrada inválida z_out y z_hat_out salen llenos de NAN (también si solo se pide uno)
            float z_out[AE_N_FEATURES];
            float z_hat_out[AE_N_FEATURES];
            memset(z_out, 0, sizeof(z_out));
            memset(z_hat_out, 0, sizeof(z_hat_out));
            check_invalid_result(__LINE__, ae_infer_ex(x, z_out, z_hat_out), what);
            CHECK(all_nan(z_out, AE_N_FEATURES) && all_nan(z_hat_out, AE_N_FEATURES),
                  "%s: ae_infer_ex debe llenar z_out y z_hat_out con NAN", what);
            memset(z_out, 0, sizeof(z_out));
            check_invalid_result(__LINE__, ae_infer_ex(x, z_out, NULL), what);
            CHECK(all_nan(z_out, AE_N_FEATURES), "%s: ae_infer_ex(x, z_out, NULL) debe llenar z_out con NAN", what);
        }
    }

    check_invalid_result(__LINE__, ae_infer(NULL), "features == NULL");

    // x finita pero z no (|x| enorme con STD < 1 desborda): ae_normalize devuelve false
    // exactamente cuando alguna z no es finita
    for (int i = 0; i < AE_N_FEATURES; i++) {
        for (int sgn = 0; sgn < 2; sgn++) {
            float x[AE_N_FEATURES];
            float z[AE_N_FEATURES];
            memcpy(x, AE_TV_X[0], sizeof(x));
            x[i] = (sgn == 0) ? FLT_MAX : -FLT_MAX;
            const bool ok = ae_normalize(x, z);
            bool z_finite = true;
            for (int j = 0; j < AE_N_FEATURES; j++) {
                z_finite = z_finite && isfinite(z[j]);
            }
            CHECK(ok == z_finite, "x[%d] = %.9g: ae_normalize devolvio %d con z %s", i, (double)x[i], (int)ok,
                  z_finite ? "finita" : "no finita");
        }
    }

    // Finito pero enorme: no debe colgarse y debe acabar en SEVERE, válido con score >= T3 o
    // inválido (+inf) si el score desborda; en ese caso z_out / z_hat_out traen lo calculado
    const float huge[3] = {1e30f, -1e30f, FLT_MAX};
    int overflow_cases = 0;
    for (int h = 0; h < 3; h++) {
        for (int p = 0; p <= 3; p++) {
            float x[AE_N_FEATURES];
            memcpy(x, AE_TV_X[0], sizeof(x));
            if (p < 3) {
                x[pos[p]] = huge[h];
            } else {
                for (int i = 0; i < AE_N_FEATURES; i++) {
                    x[i] = huge[h]; // todas las features a la vez
                }
            }
            const int where = (p < 3) ? pos[p] : -1;

            float z_out[AE_N_FEATURES];
            float z_hat_out[AE_N_FEATURES];
            const ae_result_t r = ae_infer_ex(x, z_out, z_hat_out);
            CHECK(r.level == AE_LEVEL_SEVERE, "entrada %.9g (pos %d): nivel %d, esperado SEVERE", (double)huge[h],
                  where, (int)r.level);
            CHECK(r.valid ? (isfinite(r.score) && r.score >= ae_threshold(AE_LEVEL_SEVERE)) : is_pos_inf(r.score),
                  "entrada %.9g (pos %d): valid=%d con score %.9g", (double)huge[h], where, (int)r.valid,
                  (double)r.score);

            float z[AE_N_FEATURES];
            float z_hat[AE_N_FEATURES];
            if (ae_normalize(x, z)) {
                ae_forward(z, z_hat);
                overflow_cases += r.valid ? 0 : 1;
                CHECK(memcmp(z_out, z, sizeof(z)) == 0 && memcmp(z_hat_out, z_hat, sizeof(z_hat)) == 0,
                      "entrada %.9g (pos %d): con z finita, z_out / z_hat_out deben traer los valores calculados",
                      (double)huge[h], where);
            }
        }
    }
    printf("  info: %d casos con z finita y score desbordado (valid=0, z_out / z_hat_out calculados)\n",
           overflow_cases);
}

// ---------------------------------------------------------------- (7) determinismo

static void test_determinism(void)
{
    for (int k = 0; k < AE_TV_COUNT; k++) {
        float x[AE_N_FEATURES];
        memcpy(x, AE_TV_X[k], sizeof(x));

        float z_a[AE_N_FEATURES];
        float zh_a[AE_N_FEATURES];
        float z_b[AE_N_FEATURES];
        float zh_b[AE_N_FEATURES];
        const ae_result_t a = ae_infer_ex(x, z_a, zh_a);
        (void)ae_infer(AE_TV_X[(k + 1) % AE_TV_COUNT]); // otra entrada entre medias
        const ae_result_t b = ae_infer_ex(x, z_b, zh_b);
        const ae_result_t c = ae_infer(x);

        CHECK(same_result(a, b) && same_result(a, c), "vector %d: resultados distintos en llamadas repetidas", k);
        CHECK(memcmp(z_a, z_b, sizeof(z_a)) == 0 && memcmp(zh_a, zh_b, sizeof(zh_a)) == 0,
              "vector %d: z / z_hat distintos en llamadas repetidas", k);
        CHECK(memcmp(x, AE_TV_X[k], sizeof(x)) == 0, "vector %d: la inferencia modifico la entrada", k);
    }

    // z_out puede ser el propio buffer de entrada (el motor copia al final)
    float buf[AE_N_FEATURES];
    float z_ref[AE_N_FEATURES];
    memcpy(buf, AE_TV_X[0], sizeof(buf));
    const ae_result_t ref = ae_infer_ex(AE_TV_X[0], z_ref, NULL);
    const ae_result_t alias = ae_infer_ex(buf, buf, NULL);
    CHECK(same_result(ref, alias) && memcmp(buf, z_ref, sizeof(buf)) == 0,
          "ae_infer_ex con z_out == features da otro resultado");
}

// ---------------------------------------------------------------- (8) constantes del modelo

static void test_model_constants(void)
{
    const float t1 = ae_threshold(AE_LEVEL_MILD);
    const float t2 = ae_threshold(AE_LEVEL_MODERATE);
    const float t3 = ae_threshold(AE_LEVEL_SEVERE);

    CHECK(isfinite(t1) && isfinite(t2) && isfinite(t3), "umbrales no finitos: %.9g %.9g %.9g", (double)t1,
          (double)t2, (double)t3);
    CHECK(t1 > 0.0f && t2 > 0.0f && t3 > 0.0f, "umbrales deben ser > 0: %.9g %.9g %.9g", (double)t1, (double)t2,
          (double)t3);
    CHECK(t1 <= t2 && t2 <= t3, "umbrales no crecientes: T1=%.9g T2=%.9g T3=%.9g", (double)t1, (double)t2,
          (double)t3);

    // ae_threshold: NORMAL -> 0, fuera de rango -> +inf, y cada Tk es donde empieza su nivel
    CHECK(same_bits(ae_threshold(AE_LEVEL_NORMAL), 0.0f), "ae_threshold(NORMAL) = %.9g, esperado 0",
          (double)ae_threshold(AE_LEVEL_NORMAL));
    CHECK(is_pos_inf(ae_threshold((ae_level_t)4)) && is_pos_inf(ae_threshold((ae_level_t)-1)),
          "ae_threshold(nivel invalido) debe ser +inf");
    const float ts[5] = {0.0f, t1, t2, t3, INFINITY};
    for (int lv = 1; lv <= 3; lv++) {
        const bool distinct = ts[lv] < ts[lv + 1]; // banda del nivel lv no vacía
        CHECK(!distinct || ae_level_from_score(ts[lv]) == (ae_level_t)lv,
              "ae_threshold(%d) = %.9g no es el inicio del nivel %d", lv, (double)ts[lv], lv);
        CHECK(ae_level_from_score(nextafterf(ts[lv], 0.0f)) < (ae_level_t)lv,
              "justo por debajo de ae_threshold(%d) ya es nivel %d", lv, lv);
    }

    // STD > 0 y finita (STD_VECTOR no es API pública): z[i] = (x[i] - mean[i]) / std[i] debe
    // crecer estrictamente con x[i] y no tocar las demás componentes
    float x0[AE_N_FEATURES];
    float z0[AE_N_FEATURES];
    memcpy(x0, AE_TV_X[0], sizeof(x0));
    const bool x0_ok = ae_normalize(x0, z0);
    CHECK(x0_ok, "ae_normalize rechazo AE_TV_X[0]");
    for (int i = 0; i < AE_N_FEATURES; i++) {
        float x1[AE_N_FEATURES];
        float z1[AE_N_FEATURES];
        memcpy(x1, x0, sizeof(x1));
        x1[i] += 1.0f + fabsf(x1[i]);
        (void)ae_normalize(x1, z1);
        CHECK(z1[i] > z0[i], "feature %d: z no crece con x (STD_VECTOR[%d] <= 0, infinita o NaN)", i, i);
        for (int j = 0; j < AE_N_FEATURES; j++) {
            CHECK(j == i || same_bits(z1[j], z0[j]), "feature %d: cambiar x[%d] altero z[%d]", i, i, j);
        }
    }
}

// ---------------------------------------------------------------- (9) benchmark (no falla nunca)

static void test_benchmark(void)
{
    volatile float sink = 0.0f; // volatile: el compilador no puede eliminar las inferencias
    for (int i = 0; i < 16; i++) {
        sink += ae_infer(AE_TV_X[i % AE_TV_COUNT]).score; // calentar caches
    }
    const int64_t t0 = now_us();
    for (int i = 0; i < BENCH_ITERS; i++) {
        sink += ae_infer(AE_TV_X[i % AE_TV_COUNT]).score;
    }
    const int64_t t1 = now_us();
    printf("  info: benchmark %d x ae_infer: %.3f us/inferencia (sink %.3g)\n", BENCH_ITERS,
           (double)(t1 - t0) / BENCH_ITERS, (double)sink);
}

// ---------------------------------------------------------------- runner

typedef struct {
    void (*fn)(void);
    const char *name;
    int line;
} test_case_t;

#define TEST_ENTRY(fn) {fn, #fn, __LINE__}

static const test_case_t TESTS[] = {
    TEST_ENTRY(test_model_id),
    TEST_ENTRY(test_golden_vectors),
    TEST_ENTRY(test_golden_coverage),
    TEST_ENTRY(test_activations),
    TEST_ENTRY(test_dense),
    TEST_ENTRY(test_level_boundaries),
    TEST_ENTRY(test_fail_safe),
    TEST_ENTRY(test_determinism),
    TEST_ENTRY(test_model_constants),
    TEST_ENTRY(test_benchmark),
};

static int run_all_tests(void)
{
    g_file = base_name(__FILE__);
    printf("\nae_test: modelo %s, %d vectores dorados\n", ae_model_id(), AE_TV_COUNT);
    if (ae_model_is_placeholder()) {
        printf("!!!!!!!! AVISO: PESOS PLACEHOLDER (fixture aleatorio), NO es un modelo entrenado !!!!!!!!\n");
    }

    const int n_tests = (int)(sizeof(TESTS) / sizeof(TESTS[0]));
    int failures = 0;
    for (int t = 0; t < n_tests; t++) {
        g_test_name = TESTS[t].name;
        g_fails = 0;
        TESTS[t].fn(); // un fallo imprime ya su línea FAIL (check_failed)
        if (g_fails == 0) {
            printf("%s:%d:%s:PASS\n", g_file, TESTS[t].line, TESTS[t].name);
        } else {
            failures++;
            if (g_fails > MAX_DETAIL_LINES) {
                printf("  ... %d fallos en total en %s\n", g_fails, TESTS[t].name);
            }
        }
    }
    printf("-----------------------\n");
    printf("%d Tests %d Failures 0 Ignored\n", n_tests, failures);
    printf("%s\n", failures == 0 ? "OK" : "FAIL");
    fflush(stdout);
    return failures;
}

#ifdef ESP_PLATFORM
static void test_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(1000)); // dar tiempo a abrir el monitor serie
    run_all_tests();
    vTaskDelete(NULL);
}

void app_main(void)
{
    // Tarea propia: la pila de main (3.5 KB por defecto) es justa para printf + buffers
    xTaskCreate(test_task, "ae_test", 16384, NULL, 5, NULL);
}
#else
int main(void)
{
    return run_all_tests() == 0 ? 0 : 1;
}
#endif
