// Test del motor del ErrP-AE (src/autoencoder_engine.c + runner TFLM) con los vectores dorados
// del exportador (autoencoder_test_vectors.h, en este mismo directorio) más casos unitarios.
// CONTRACT v3, sección 8.
//
// Niveles:
//   A (PC, siempre): -DAE_TEST_FAKE_RUNNER. El test aporta ae_runner_*: el runner falso
//     devuelve el q_out dorado cuando q_in coincide EXACTAMENTE con un q_in dorado (si no,
//     falla), permite inyectar fallos (init, invoke, metadatos corruptos) y tiene un modo "eco"
//     (q_out = q_in) solo para probar el desbordamiento del score. Toma los metadatos de
//     include/autoencoder_weights.h.
//   B (PC con TFLite Micro real, kernels de referencia, sin ESP-NN): el mismo test enlazado con
//     src/autoencoder_tflm.cc: q_out debe ser bit a bit el dorado; imprime la arena usada.
//     Los q_out dorados deben replicar la aritmética de TFLM: tf.lite.Interpreter con
//     OpResolverType.BUILTIN_REF (TF 2.21) coincide en CONV_2D, DEPTHWISE_CONV_2D y
//     AVERAGE_POOL_2D, pero recuantiza FULLY_CONNECTED con redondeo simple (TFLM: doble redondeo
//     gemmlowp) y en los empates sale 1 LSB distinto, diferencia que se propaga por las capas
//     densas siguientes. Por eso export_to_c.py genera por defecto los q_out con su emulación
//     entera de TFLM (--golden-runtime tflm, verificada bit a bit contra TFLM real).
//   C (ESP32-S3, ESP-IDF define ESP_PLATFORM): app_main corre la misma batería con el runner
//     real (ESP-NN) en una tarea de 16 KB y el benchmark exige < 20 ms por época (CLAUDE.md,
//     test 12). Necesita esp_timer en REQUIRES/PRIV_REQUIRES.
//
// Contenido que usa de autoencoder_test_vectors.h (GENERADO por ml/c_exporter/export_to_c.py):
//   #define AE_TV_MODEL_ID "<16 hex>" (== AE_MODEL_ID), AE_TV_N_WIN, AE_TV_COUNT, AE_TV_CAL_N_EPOCHS,
//           AE_TV_CAL_N_SCORES (>= AE_CALIB_MIN_SCORES)
//   float  AE_TV_WIN[AE_TV_N_WIN][8][250], AE_TV_WIN_EPOCH[AE_TV_N_WIN][8][40]   preprocesado
//   float  AE_TV_EPOCH[AE_TV_COUNT][8][40], AE_TV_Z[AE_TV_COUNT][320], AE_TV_ZHAT[AE_TV_COUNT][320],
//          AE_TV_SCORE[AE_TV_COUNT]                    (calibración por defecto del modelo)
//   int8_t AE_TV_Q_IN[AE_TV_COUNT][320], AE_TV_Q_OUT[AE_TV_COUNT][320]
//   int    AE_TV_LEVEL[AE_TV_COUNT]; unsigned char AE_TV_NEAR_THRESHOLD[AE_TV_COUNT] (1: no exigir nivel)
//   float  AE_TV_CAL_EPOCHS[AE_TV_CAL_N_EPOCHS][8][40], AE_TV_CAL_MEAN[8], AE_TV_CAL_STD[8]  (ddof 0)
//   float  AE_TV_CAL_SCORES[AE_TV_CAL_N_SCORES], AE_TV_CAL_T[3]   (percentiles AE_CALIB_PCT_1..3)
// Las épocas pueden ir como [N][8][40] o [N][320] (mismo layout); los _Static_assert de abajo
// comprueban tamaños y cuentas.
//
// Harness propio (sin Unity) con salida compatible con Unity / PlatformIO:
//   test_c_engine.c:<línea>:<test>:PASS
//   test_c_engine.c:<línea>:<test>:FAIL: <mensaje>
//   -----------------------
//   <N> Tests <M> Failures 0 Ignored
//   OK | FAIL
// Como un TEST_ASSERT de Unity, el primer CHECK fallido de un test imprime su línea FAIL; los
// siguientes fallos (hasta MAX_DETAIL_LINES) salen debajo, indentados, como detalle.
//
// Nivel A, desde la raíz del repo (CC = gcc en Linux, "python -m ziglang cc" en Windows; zig
// respeta -o tal cual: ./test_c_engine funciona en Git Bash, en PowerShell usar .exe):
//   $CC -std=c11 -O2 -Wall -Wextra -Wpedantic -Wdouble-promotion -Wfloat-conversion -Wshadow -Werror -DAE_TEST_FAKE_RUNNER -I firmware/detector_s3/include firmware/detector_s3/src/autoencoder_engine.c firmware/detector_s3/test/test_c_engine.c -lm -o test_c_engine
//   ./test_c_engine            (código de salida != 0 si falla algún test)
// (con -O0 -g, zig cc añade además UBSan con trap).
//
// Nivel B: 1) TFLite Micro para el PC desde espressif/esp-tflite-micro 1.4.1 (TFLM = su raíz),
// kernels de referencia, sin ESP-NN ni archivos esp/, en una biblioteca estática (solo el
// intérprete y las 6 ops; ~30 s con 8 compilaciones en paralelo):
//   CXX="python -m ziglang c++"
//   INC="-I $TFLM -I $TFLM/third_party/flatbuffers/include -I $TFLM/third_party/gemmlowp -I $TFLM/third_party/ruy"
//   ISYS="-isystem $TFLM -isystem $TFLM/third_party/flatbuffers/include -isystem $TFLM/third_party/gemmlowp -isystem $TFLM/third_party/ruy"
//   L=$TFLM/tensorflow/lite; M=$L/micro; K=$M/kernels
//   SRCS="$M/debug_log.cc $M/flatbuffer_utils.cc $M/memory_helpers.cc $M/micro_allocation_info.cc
//     $M/micro_allocator.cc $M/micro_context.cc $M/micro_interpreter_context.cc $M/micro_interpreter_graph.cc
//     $M/micro_interpreter.cc $M/micro_log.cc $M/micro_op_resolver.cc $M/micro_profiler.cc
//     $M/micro_resource_variable.cc $M/micro_time.cc $M/micro_utils.cc $M/system_setup.cc
//     $M/tflite_bridge/flatbuffer_conversions_bridge.cc $M/tflite_bridge/micro_error_reporter.cc
//     $M/arena_allocator/non_persistent_arena_buffer_allocator.cc $M/arena_allocator/persistent_arena_buffer_allocator.cc
//     $M/arena_allocator/single_arena_buffer_allocator.cc $M/memory_planner/greedy_memory_planner.cc
//     $M/memory_planner/linear_memory_planner.cc
//     $K/activations.cc $K/activations_common.cc $K/conv.cc $K/conv_common.cc $K/depthwise_conv.cc
//     $K/depthwise_conv_common.cc $K/fully_connected.cc $K/fully_connected_common.cc $K/pooling.cc
//     $K/pooling_common.cc $K/reshape.cc $K/reshape_common.cc $K/kernel_util.cc
//     $L/core/c/common.cc $L/core/api/flatbuffer_conversions.cc $L/core/api/tensor_utils.cc
//     $L/kernels/kernel_util.cc $L/kernels/internal/common.cc $L/kernels/internal/quantization_util.cc
//     $L/kernels/internal/portable_tensor_utils.cc $L/kernels/internal/tensor_utils.cc
//     $L/kernels/internal/tensor_ctypes.cc $L/kernels/internal/reference/portable_tensor_utils.cc
//     $TFLM/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc $TFLM/tensorflow/compiler/mlir/lite/schema/schema_utils.cc"
//   mkdir -p obj; for f in $SRCS; do o=obj/$(echo "${f#$TFLM/}" | tr / _).o; $CXX -std=c++17 -O2 -fno-exceptions -fno-rtti -ffp-contract=off -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -w $INC -c $f -o $o; done
//   python -m ziglang ar rcs libtflm_host.a obj/*.o
// 2) motor, runner y test (mismas banderas estrictas que el nivel A, sin -DAE_TEST_FAKE_RUNNER):
//   $CC -std=c11 -O2 -Wall -Wextra -Wpedantic -Wdouble-promotion -Wfloat-conversion -Wshadow -Werror -I firmware/detector_s3/include -c firmware/detector_s3/src/autoencoder_engine.c -o ae_engine.o
//   $CC <las mismas banderas> -I firmware/detector_s3/include -c firmware/detector_s3/test/test_c_engine.c -o ae_test.o
//   $CXX -std=c++17 -O2 -fno-exceptions -fno-rtti -DTF_LITE_STATIC_MEMORY -Wall -Wextra -Werror -I firmware/detector_s3/include $ISYS -c firmware/detector_s3/src/autoencoder_tflm.cc -o ae_tflm.o
//   $CXX ae_test.o ae_engine.o ae_tflm.o libtflm_host.a -o test_c_engine_tflm && ./test_c_engine_tflm
// Con "-target x86-windows-gnu" en todos los pasos se mide la arena con punteros de 32 bits,
// como en el S3 (la de 64 bits es una cota superior; ESP-NN puede pedir scratch adicional).
//
// Si falla test_init por AE_MODEL_ID, pesos y vectores son de exportaciones distintas:
// regenerar los dos a la vez con ml/c_exporter/export_to_c.py.
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "autoencoder_engine.h"
#include "autoencoder_test_vectors.h"

#ifdef AE_TEST_FAKE_RUNNER
#include "autoencoder_weights.h" // metadatos del modelo para el runner falso (solo nivel A)
#endif

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#define BENCH_ITERS 200
#else
#include <time.h>
#define BENCH_ITERS 2000
#endif

// Presupuesto del CLAUDE.md (test 12): < 20 ms por época en el S3
#define AE_BUDGET_US 20000.0

#ifdef AE_TEST_FAKE_RUNNER
static const char *const RUNNER_KIND = "runner falso: nivel A, sin TFLM";
#else
static const char *const RUNNER_KIND = "runner TFLM real";
#endif

// ---------------------------------------------------------------- layout de los vectores dorados
// Las épocas pueden venir como [N][8][40] o [N][320] (mismo layout en memoria): se leen con
// los accesores tv_*() y estos asserts fijan tamaños y cuentas.

#define TV_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

_Static_assert(AE_TV_COUNT >= 1 && AE_TV_N_WIN >= 1 && AE_TV_CAL_N_EPOCHS >= 1, "autoencoder_test_vectors.h vacío");
_Static_assert(AE_TV_CAL_N_SCORES >= AE_CALIB_MIN_SCORES, "AE_TV_CAL_SCORES necesita >= AE_CALIB_MIN_SCORES scores");
_Static_assert(sizeof(AE_TV_WIN[0]) == sizeof(float) * AE_N_CH * AE_WIN_SAMPLES, "AE_TV_WIN: 8 x 250 por ventana");
_Static_assert(sizeof(AE_TV_WIN_EPOCH[0]) == sizeof(float) * AE_N_IN, "AE_TV_WIN_EPOCH: 8 x 40 por ventana");
_Static_assert(sizeof(AE_TV_EPOCH[0]) == sizeof(float) * AE_N_IN, "AE_TV_EPOCH: 8 x 40 por vector");
_Static_assert(sizeof(AE_TV_Z[0]) == sizeof(float) * AE_N_IN, "AE_TV_Z: 320 por vector");
_Static_assert(sizeof(AE_TV_Q_IN[0]) == AE_N_IN, "AE_TV_Q_IN: 320 int8 por vector");
_Static_assert(sizeof(AE_TV_Q_OUT[0]) == AE_N_IN, "AE_TV_Q_OUT: 320 int8 por vector");
_Static_assert(sizeof(AE_TV_ZHAT[0]) == sizeof(float) * AE_N_IN, "AE_TV_ZHAT: 320 por vector");
_Static_assert(sizeof(AE_TV_CAL_EPOCHS[0]) == sizeof(float) * AE_N_IN, "AE_TV_CAL_EPOCHS: 8 x 40 por época");
_Static_assert(TV_LEN(AE_TV_WIN) == AE_TV_N_WIN && TV_LEN(AE_TV_WIN_EPOCH) == AE_TV_N_WIN, "cuenta de ventanas");
_Static_assert(TV_LEN(AE_TV_EPOCH) == AE_TV_COUNT && TV_LEN(AE_TV_Z) == AE_TV_COUNT &&
                   TV_LEN(AE_TV_Q_IN) == AE_TV_COUNT && TV_LEN(AE_TV_Q_OUT) == AE_TV_COUNT &&
                   TV_LEN(AE_TV_ZHAT) == AE_TV_COUNT && TV_LEN(AE_TV_SCORE) == AE_TV_COUNT &&
                   TV_LEN(AE_TV_LEVEL) == AE_TV_COUNT && TV_LEN(AE_TV_NEAR_THRESHOLD) == AE_TV_COUNT,
               "cuenta de vectores != AE_TV_COUNT");
_Static_assert(TV_LEN(AE_TV_CAL_EPOCHS) == AE_TV_CAL_N_EPOCHS && TV_LEN(AE_TV_CAL_SCORES) == AE_TV_CAL_N_SCORES &&
                   TV_LEN(AE_TV_CAL_MEAN) == AE_N_CH && TV_LEN(AE_TV_CAL_STD) == AE_N_CH && TV_LEN(AE_TV_CAL_T) == 3,
               "cuentas de los goldens de calibración");

typedef float ae_row_t[AE_N_T];            // una fila (canal) de una época
typedef float ae_win_row_t[AE_WIN_SAMPLES]; // una fila (canal) de una ventana cruda

static const ae_row_t *tv_epoch(int k)
{
    return (const ae_row_t *)(const void *)AE_TV_EPOCH[k];
}

static const ae_row_t *tv_cal_epoch(int k)
{
    return (const ae_row_t *)(const void *)AE_TV_CAL_EPOCHS[k];
}

static const ae_win_row_t *tv_win(int w)
{
    return (const ae_win_row_t *)(const void *)AE_TV_WIN[w];
}

static const float *tv_win_epoch(int w)
{
    return (const float *)(const void *)AE_TV_WIN_EPOCH[w];
}

static const float *tv_z(int k)
{
    return (const float *)(const void *)AE_TV_Z[k];
}

static const int8_t *tv_q_in(int k)
{
    return (const int8_t *)(const void *)AE_TV_Q_IN[k];
}

static const int8_t *tv_q_out(int k)
{
    return (const int8_t *)(const void *)AE_TV_Q_OUT[k];
}

static const float *tv_zhat(int k)
{
    return (const float *)(const void *)AE_TV_ZHAT[k];
}

// Época o ventana local (no const) -> parámetro const (C11 + -Wpedantic lo exige explícito)
#define CEPOCH(e) ((const ae_row_t *)(e))
#define CWIN(w) ((const ae_win_row_t *)(w))

// Tolerancias. El score se compara con 1e-6 relativo (contrato): con FMA (madd.s en el S3 o
// -ffp-contract en el PC) la suma puede diferir ~1 ulp; todo lo demás de la cadena es bit a bit.
#define TOL_SCORE_REL 1e-6f
// Calibración: Welford float32 (C) frente a numpy en float64. 2e-5 relativo deja margen al
// redondeo float32 y detecta ddof 1 (cambia la std en 1/(2n): 2e-3 con 6 épocas = 240 muestras).
#define TOL_STD_REL 2e-5f
// Percentiles: float32 frente a float64 de numpy (pocos ulp)
#define TOL_PCT_REL 1e-6f

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
    char msg[320];
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

static bool near_rel(float actual, float expected, float rel, float abs_tol)
{
    return fabsf(actual - expected) <= abs_tol + rel * fabsf(expected); // NaN -> false
}

// Distancia en ulps entre dos float finitos (orden entero monótono)
static uint32_t ulp_diff(float a, float b)
{
    int32_t ia;
    int32_t ib;
    memcpy(&ia, &a, sizeof(ia));
    memcpy(&ib, &b, sizeof(ib));
    if (ia < 0) {
        ia = (int32_t)(0x80000000u - (uint32_t)ia);
    }
    if (ib < 0) {
        ib = (int32_t)(0x80000000u - (uint32_t)ib);
    }
    const int64_t d = (int64_t)ia - (int64_t)ib;
    return (uint32_t)(d < 0 ? -d : d);
}

// Número de posiciones con bits distintos; *first = la primera (o -1)
static int float_mismatches(const float *a, const float *b, int n, int *first)
{
    int count = 0;
    *first = -1;
    for (int i = 0; i < n; i++) {
        if (!same_bits(a[i], b[i])) {
            if (count == 0) {
                *first = i;
            }
            count++;
        }
    }
    return count;
}

static int all_nan(const float *v, int n)
{
    for (int i = 0; i < n; i++) {
        if (!isnan(v[i])) {
            return 0;
        }
    }
    return 1;
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

// Generador congruencial determinista (mismos datos en el PC y en el S3)
static uint32_t g_rng = 12345u;

static float rnd_uniform(void) // [0, 1)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)(g_rng >> 8) * (1.0f / 16777216.0f);
}

static float rnd_normal(void) // aproximación de N(0, 1): suma de 12 uniformes
{
    float s = 0.0f;
    for (int i = 0; i < 12; i++) {
        s += rnd_uniform();
    }
    return s - 6.0f;
}

static void set_model_id(ae_calib_t *c, const char *id)
{
    memset(c->model_id, 0, sizeof(c->model_id));
    for (int i = 0; i < AE_MODEL_ID_LEN && id[i] != '\0'; i++) {
        c->model_id[i] = id[i];
    }
}

static void make_calib(ae_calib_t *c, float mean, float std, float t1, float t2, float t3, const char *id)
{
    memset(c, 0, sizeof(*c));
    for (int ch = 0; ch < AE_N_CH; ch++) {
        c->mean[ch] = mean;
        c->std[ch] = std;
    }
    c->t1 = t1;
    c->t2 = t2;
    c->t3 = t3;
    set_model_id(c, id);
}

// Workspaces estáticos: la pila de la tarea de test del S3 no tiene que alojarlos
static ae_workspace_t g_ws;
static ae_workspace_t g_ws2;

// ---------------------------------------------------------------- runner falso (nivel A)

#ifdef AE_TEST_FAKE_RUNNER

enum {
    FAKE_OK = 0,
    FAKE_ID_UPPER,
    FAKE_ID_SHORT,
    FAKE_ID_NULL,
    FAKE_ARCH_NULL,
    FAKE_ARCH_EMPTY,
    FAKE_IN_SCALE_ZERO,
    FAKE_IN_SCALE_NAN,
    FAKE_OUT_SCALE_NEG,
    FAKE_OUT_SCALE_INF,
    FAKE_IN_ZP_128,
    FAKE_OUT_ZP_M129,
    FAKE_DEF_STD_ZERO,
    FAKE_DEF_MEAN_NAN,
    FAKE_DEF_T_UNORDERED,
    FAKE_DEF_T1_ZERO,
    FAKE_PCT_ABOVE_100,
    FAKE_PCT_ZERO,
    FAKE_PCT_UNORDERED,
    FAKE_ARENA_OVER,
    FAKE_ARENA_ZERO,
    FAKE_N_CORRUPT
};

static const char *const FAKE_CORRUPT_NAME[FAKE_N_CORRUPT] = {
    "ok", "model_id en mayusculas", "model_id corto", "model_id NULL", "arch NULL", "arch vacio",
    "in_scale 0", "in_scale NaN", "out_scale negativo", "out_scale inf", "in_zero_point 128",
    "out_zero_point -129", "default_std 0", "default_mean NaN", "umbrales por defecto desordenados",
    "T1 por defecto 0", "percentil > 100", "percentil 0", "percentiles desordenados",
    "arena usada > arena", "arena 0",
};

typedef struct {
    bool fail_init;   // ae_runner_init devuelve false
    bool fail_invoke; // ae_runner_invoke devuelve false
    bool echo;        // q_out = q_in para cualquier entrada (solo tests de desbordamiento del score)
    int corrupt;      // FAKE_*: metadatos corruptos que ae_init debe rechazar
    int n_init;
    int n_invoke;
    int n_miss; // q_in sin golden exacto
} fake_runner_t;

static fake_runner_t g_fake;

static void fake_corrupt(ae_model_info_t *info, int what)
{
    switch (what) {
    case FAKE_ID_UPPER:
        info->model_id = "0123456789ABCDEF";
        break;
    case FAKE_ID_SHORT:
        info->model_id = "0123456789abcde";
        break;
    case FAKE_ID_NULL:
        info->model_id = NULL;
        break;
    case FAKE_ARCH_NULL:
        info->arch = NULL;
        break;
    case FAKE_ARCH_EMPTY:
        info->arch = "";
        break;
    case FAKE_IN_SCALE_ZERO:
        info->in_scale = 0.0f;
        break;
    case FAKE_IN_SCALE_NAN:
        info->in_scale = NAN;
        break;
    case FAKE_OUT_SCALE_NEG:
        info->out_scale = -info->out_scale;
        break;
    case FAKE_OUT_SCALE_INF:
        info->out_scale = INFINITY;
        break;
    case FAKE_IN_ZP_128:
        info->in_zero_point = 128;
        break;
    case FAKE_OUT_ZP_M129:
        info->out_zero_point = -129;
        break;
    case FAKE_DEF_STD_ZERO:
        info->default_std[3] = 0.0f;
        break;
    case FAKE_DEF_MEAN_NAN:
        info->default_mean[0] = NAN;
        break;
    case FAKE_DEF_T_UNORDERED:
        info->default_t[0] = info->default_t[2] * 2.0f;
        break;
    case FAKE_DEF_T1_ZERO:
        info->default_t[0] = 0.0f;
        break;
    case FAKE_PCT_ABOVE_100:
        info->calib_pct[2] = 100.5f;
        break;
    case FAKE_PCT_ZERO:
        info->calib_pct[0] = 0.0f;
        break;
    case FAKE_PCT_UNORDERED:
        info->calib_pct[1] = info->calib_pct[2] + 0.5f;
        break;
    case FAKE_ARENA_OVER:
        info->arena_used_bytes = info->arena_bytes + 1u;
        break;
    case FAKE_ARENA_ZERO:
        info->arena_bytes = 0u;
        break;
    default:
        break;
    }
}

bool ae_runner_init(ae_model_info_t *info)
{
    g_fake.n_init++;
    if (info == NULL || g_fake.fail_init) {
        return false;
    }
    info->model_id = AE_MODEL_ID;
    info->arch = AE_MODEL_ARCH;
    info->placeholder = (AE_WEIGHTS_PLACEHOLDER) != 0;
    info->in_scale = AE_IN_SCALE;
    info->in_zero_point = AE_IN_ZERO_POINT;
    info->out_scale = AE_OUT_SCALE;
    info->out_zero_point = AE_OUT_ZERO_POINT;
    memcpy(info->default_mean, MEAN_VECTOR, sizeof(info->default_mean));
    memcpy(info->default_std, STD_VECTOR, sizeof(info->default_std));
    info->default_t[0] = THRESHOLD_LEVEL_1;
    info->default_t[1] = THRESHOLD_LEVEL_2;
    info->default_t[2] = THRESHOLD_LEVEL_3;
    info->calib_pct[0] = AE_CALIB_PCT_1;
    info->calib_pct[1] = AE_CALIB_PCT_2;
    info->calib_pct[2] = AE_CALIB_PCT_3;
    info->arena_bytes = (uint32_t)(AE_TENSOR_ARENA_BYTES);
    info->arena_used_bytes = 0u; // sin TFLM no hay arena
    info->int8_score_corr = AE_INT8_SCORE_CORR;
    fake_corrupt(info, g_fake.corrupt);
    return true;
}

bool ae_runner_invoke(const int8_t in[AE_N_IN], int8_t out[AE_N_IN])
{
    g_fake.n_invoke++;
    if (g_fake.fail_invoke || in == NULL || out == NULL) {
        return false;
    }
    if (g_fake.echo) {
        memcpy(out, in, AE_N_IN);
        return true;
    }
    for (int k = 0; k < AE_TV_COUNT; k++) {
        if (memcmp(in, tv_q_in(k), AE_N_IN) == 0) {
            memcpy(out, tv_q_out(k), AE_N_IN);
            return true;
        }
    }
    g_fake.n_miss++;
    return false;
}

#endif // AE_TEST_FAKE_RUNNER

// ---------------------------------------------------------------- (1) sin inicializar (va primero)

static void test_not_initialised(void)
{
    CHECK(!ae_is_ready(), "ae_is_ready() = true antes de ae_init");
    const ae_model_info_t *info = ae_model_info();
    CHECK(info != NULL && info->model_id != NULL && info->model_id[0] == '\0' && info->arch != NULL &&
              info->arch[0] == '\0',
          "ae_model_info() sin inicializar debe dar model_id = arch = \"\"");

    ae_calib_t def;
    ae_calib_default(&def);
    CHECK(!ae_calib_check(&def) && def.model_id[0] == '\0' && isnan(def.t1) && isnan(def.std[0]),
          "ae_calib_default sin inicializar debe dar una calibracion invalida");

    // Ni siquiera una calibración plausible con el id de los vectores vale sin modelo
    ae_calib_t c;
    make_calib(&c, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, AE_TV_MODEL_ID);
    CHECK(!ae_calib_check(&c), "ae_calib_check = true sin ae_init");
    check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &c, &g_ws), "ae_infer sin ae_init");

    float scores[AE_CALIB_MIN_SCORES];
    for (int i = 0; i < AE_CALIB_MIN_SCORES; i++) {
        scores[i] = 1.0f + (float)i;
    }
    CHECK(!ae_calib_thresholds_from_scores(&c, scores, AE_CALIB_MIN_SCORES) && isnan(c.t1) && isnan(c.t3),
          "ae_calib_thresholds_from_scores sin ae_init debe fallar y dejar umbrales NaN");
#ifdef AE_TEST_FAKE_RUNNER
    CHECK(g_fake.n_invoke == 0, "el runner se invoco %d veces sin ae_init", g_fake.n_invoke);
#endif
}

// ---------------------------------------------------------------- (2) ae_init y metadatos

static bool is_lower_hex16(const char *id)
{
    if (id == NULL) {
        return false;
    }
    for (int i = 0; i < AE_MODEL_ID_LEN; i++) {
        if (!((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f'))) {
            return false;
        }
    }
    return id[AE_MODEL_ID_LEN] == '\0';
}

static void test_init(void)
{
    const bool ok = ae_init();
    CHECK(ok && ae_is_ready(), "ae_init() fallo con el modelo compilado (ver mensajes ae_tflm)");
    if (!ok) {
        return;
    }
    const ae_model_info_t *info = ae_model_info();
    CHECK(strcmp(info->model_id, AE_TV_MODEL_ID) == 0,
          "AE_MODEL_ID del modelo (%s) != AE_TV_MODEL_ID de los vectores (%s): regenerar ambos con "
          "ml/c_exporter/export_to_c.py",
          info->model_id, AE_TV_MODEL_ID);
    CHECK(is_lower_hex16(info->model_id), "AE_MODEL_ID \"%s\" no son 16 hex en minusculas", info->model_id);
    CHECK(strcmp(info->arch, "errp_conv") == 0 || strcmp(info->arch, "dense") == 0,
          "AE_MODEL_ARCH \"%s\" no es errp_conv ni dense", info->arch);
    CHECK(isfinite(info->in_scale) && info->in_scale > 0.0f && isfinite(info->out_scale) && info->out_scale > 0.0f,
          "escalas de cuantizacion invalidas: in %.9g out %.9g", (double)info->in_scale, (double)info->out_scale);
    CHECK(info->in_zero_point >= -128 && info->in_zero_point <= 127 && info->out_zero_point >= -128 &&
              info->out_zero_point <= 127,
          "zero points fuera de int8: in %d out %d", (int)info->in_zero_point, (int)info->out_zero_point);
    CHECK(info->default_t[0] > 0.0f && info->default_t[0] <= info->default_t[1] &&
              info->default_t[1] <= info->default_t[2] && isfinite(info->default_t[2]),
          "umbrales por defecto invalidos: %.9g %.9g %.9g", (double)info->default_t[0], (double)info->default_t[1],
          (double)info->default_t[2]);
    CHECK(info->calib_pct[0] > 0.0f && info->calib_pct[0] <= info->calib_pct[1] &&
              info->calib_pct[1] <= info->calib_pct[2] && info->calib_pct[2] <= 100.0f,
          "percentiles de calibracion invalidos: %.9g %.9g %.9g", (double)info->calib_pct[0],
          (double)info->calib_pct[1], (double)info->calib_pct[2]);
    CHECK(info->arena_bytes > 0u && info->arena_used_bytes <= info->arena_bytes, "arena usada %u > arena %u B",
          (unsigned)info->arena_used_bytes, (unsigned)info->arena_bytes);
#ifndef AE_TEST_FAKE_RUNNER
    CHECK(info->arena_used_bytes > 0u, "arena usada = 0 con el runner TFLM real");
#endif

    ae_calib_t c;
    ae_calib_default(&c);
    CHECK(ae_calib_check(&c), "la calibracion por defecto no pasa ae_calib_check");
    CHECK(memcmp(c.mean, info->default_mean, sizeof(c.mean)) == 0 &&
              memcmp(c.std, info->default_std, sizeof(c.std)) == 0 && same_bits(c.t1, info->default_t[0]) &&
              same_bits(c.t2, info->default_t[1]) && same_bits(c.t3, info->default_t[2]) &&
              strcmp(c.model_id, info->model_id) == 0,
          "ae_calib_default no copia los valores por defecto del modelo");

#ifdef AE_TEST_FAKE_RUNNER
    // Header de pesos (solo se incluye en el nivel A): el modelo es un flatbuffer TFLite
    CHECK(AE_MODEL_TFLITE_LEN > 8u && memcmp(&AE_MODEL_TFLITE[4], "TFL3", 4) == 0,
          "AE_MODEL_TFLITE no es un flatbuffer TFLite (falta el identificador TFL3)");
#endif

    printf("  info: modelo %s (%s%s), entrada s=%.9g zp=%d, salida s=%.9g zp=%d, T=%.6g/%.6g/%.6g "
           "(p%g/p%g/p%g), arena usada %u de %u B (%s), corr int8 %.4f\n",
           info->model_id, info->arch, info->placeholder ? ", PLACEHOLDER" : "", (double)info->in_scale,
           (int)info->in_zero_point, (double)info->out_scale, (int)info->out_zero_point, (double)info->default_t[0],
           (double)info->default_t[1], (double)info->default_t[2], (double)info->calib_pct[0],
           (double)info->calib_pct[1], (double)info->calib_pct[2], (unsigned)info->arena_used_bytes,
           (unsigned)info->arena_bytes, RUNNER_KIND, (double)info->int8_score_corr);
    if (info->placeholder) {
        printf("!!!!!!!! AVISO: PESOS PLACEHOLDER (fixture aleatorio), NO es un modelo entrenado !!!!!!!!\n");
    }

    // Otra llamada a ae_init reconstruye el intérprete y da exactamente los mismos resultados
    const ae_result_t before = ae_infer(tv_epoch(0), &c, &g_ws);
    CHECK(ae_init() && ae_is_ready(), "la segunda llamada a ae_init fallo");
    const ae_result_t after = ae_infer(tv_epoch(0), &c, &g_ws);
    CHECK(before.valid && same_result(before, after), "resultado distinto tras repetir ae_init");
}

// ---------------------------------------------------------------- (3) preprocesado

static void test_preprocess_goldens(void)
{
    float ep[AE_N_CH][AE_N_T];
    int total = 0;
    for (int w = 0; w < AE_TV_N_WIN; w++) {
        ae_preprocess(tv_win(w), ep);
        int first;
        const int bad = float_mismatches(&ep[0][0], tv_win_epoch(w), AE_N_IN, &first);
        total += bad;
        CHECK(bad == 0, "ventana %d: %d muestras distintas del golden (primera [%d][%d]: %.9g vs %.9g)", w, bad,
              first / AE_N_T, first % AE_N_T, (double)(&ep[0][0])[first < 0 ? 0 : first],
              (double)tv_win_epoch(w)[first < 0 ? 0 : first]);
    }
    printf("  info: %d ventanas, %d muestras distintas (bit a bit)\n", AE_TV_N_WIN, total);
}

static void test_preprocess_structure(void)
{
    static float win[AE_N_CH][AE_WIN_SAMPLES];
    float ep[AE_N_CH][AE_N_T];

    // Rampa win[c][j] = j + 100c: base = 24.5 + 100c y media del bloque k = 52 + 5k + 100c
    // -> epoch[c][k] = 27.5 + 5k exacto (enteros pequeños y potencias de 2 en float)
    for (int ch = 0; ch < AE_N_CH; ch++) {
        for (int j = 0; j < AE_WIN_SAMPLES; j++) {
            win[ch][j] = (float)(j + 100 * ch);
        }
    }
    ae_preprocess(CWIN(win), ep);
    for (int ch = 0; ch < AE_N_CH; ch++) {
        for (int k = 0; k < AE_N_T; k++) {
            CHECK(same_bits(ep[ch][k], 27.5f + 5.0f * (float)k), "rampa: epoch[%d][%d] = %.9g, esperado %.9g", ch, k,
                  (double)ep[ch][k], (double)(27.5f + 5.0f * (float)k));
        }
    }

    // Impulso aislado en la muestra j0 de un canal: va a la línea base (j0 < 50) o solo al
    // bloque (j0 - 50) / 5. Detecta ventanas desplazadas y diezmados con desfase.
    static const int js[] = {0, 1, 48, 49, 50, 51, 53, 54, 55, 59, 60, 127, 244, 245, 248, 249};
    for (int s = 0; s < (int)(sizeof(js) / sizeof(js[0])); s++) {
        const int j0 = js[s];
        const int c0 = s % AE_N_CH;
        memset(win, 0, sizeof(win));
        win[c0][j0] = 1.0f;
        ae_preprocess(CWIN(win), ep);
        for (int ch = 0; ch < AE_N_CH; ch++) {
            for (int k = 0; k < AE_N_T; k++) {
                float expected = 0.0f;
                if (ch == c0) {
                    const float base = (j0 < AE_PRE_SAMPLES) ? 1.0f / (float)AE_PRE_SAMPLES : 0.0f;
                    const bool in_block = j0 >= AE_PRE_SAMPLES && (j0 - AE_PRE_SAMPLES) / AE_DECIM == k;
                    expected = (in_block ? 1.0f : 0.0f) / (float)AE_DECIM - base;
                }
                CHECK(same_bits(ep[ch][k], expected), "impulso en win[%d][%d]: epoch[%d][%d] = %.9g, esperado %.9g",
                      c0, j0, ch, k, (double)ep[ch][k], (double)expected);
            }
        }
    }

    // NaN en la línea base contamina todo su canal; en un bloque, solo esa muestra
    memset(win, 0, sizeof(win));
    win[2][10] = NAN;
    win[5][50 + 5 * 7 + 3] = NAN;
    ae_preprocess(CWIN(win), ep);
    CHECK(all_nan(ep[2], AE_N_T), "NaN en la linea base del canal 2 no contamino todo el canal");
    for (int k = 0; k < AE_N_T; k++) {
        CHECK(k == 7 ? isnan(ep[5][k]) : same_bits(ep[5][k], 0.0f),
              "NaN en el bloque 7 del canal 5: epoch[5][%d] = %.9g", k, (double)ep[5][k]);
    }

    // win == NULL -> época llena de NaN (y ae_infer la rechaza)
    ae_preprocess(NULL, ep);
    CHECK(all_nan(&ep[0][0], AE_N_IN), "ae_preprocess(NULL, epoch) debe llenar epoch de NaN");
    ae_preprocess(CWIN(win), NULL); // no debe colgarse
}

// ---------------------------------------------------------------- (4) normalización

static void test_normalize(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    float z[AE_N_IN];
    int total = 0;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        const bool ok = ae_normalize(tv_epoch(k), &def, z);
        int first;
        const int bad = float_mismatches(z, tv_z(k), AE_N_IN, &first);
        total += bad;
        CHECK(ok, "vector %d: ae_normalize rechazo una epoca valida", k);
        CHECK(bad == 0, "vector %d: %d z distintas del golden (primera %d: %.9g vs %.9g)", k, bad, first,
              (double)z[first < 0 ? 0 : first], (double)tv_z(k)[first < 0 ? 0 : first]);
    }
    printf("  info: %d vectores, %d z distintas (bit a bit)\n", AE_TV_COUNT, total);

    // Layout canal-mayor (z[c*40 + t]) y división por std (std = 3: no es potencia de 2, así
    // que multiplicar por std o por 1/std da otros bits)
    ae_calib_t c;
    make_calib(&c, 0.0f, 3.0f, 1.0f, 2.0f, 3.0f, def.model_id);
    float ep[AE_N_CH][AE_N_T];
    for (int ch = 0; ch < AE_N_CH; ch++) {
        c.mean[ch] = (float)ch * 0.5f;
        for (int t = 0; t < AE_N_T; t++) {
            ep[ch][t] = (float)(ch * 1000 + t) * 0.37f;
        }
    }
    CHECK(ae_normalize(CEPOCH(ep), &c, z), "ae_normalize rechazo una epoca sintetica valida");
    for (int ch = 0; ch < AE_N_CH; ch++) {
        for (int t = 0; t < AE_N_T; t++) {
            const float expected = (ep[ch][t] - c.mean[ch]) / c.std[ch];
            CHECK(same_bits(z[ch * AE_N_T + t], expected), "z[%d] (canal %d, t %d) = %.9g, esperado %.9g",
                  ch * AE_N_T + t, ch, t, (double)z[ch * AE_N_T + t], (double)expected);
        }
    }

    // No finitos: false, pero z se escribe entera (las posiciones finitas con su valor)
    const float bad_vals[3] = {NAN, INFINITY, -INFINITY};
    for (int b = 0; b < 3; b++) {
        float ep2[AE_N_CH][AE_N_T];
        memcpy(ep2, ep, sizeof(ep2));
        ep2[4][20] = bad_vals[b];
        float z2[AE_N_IN];
        CHECK(!ae_normalize(CEPOCH(ep2), &c, z2), "ae_normalize acepto %.9g", (double)bad_vals[b]);
        CHECK(same_bits(z2[0], z[0]) && same_bits(z2[AE_N_IN - 1], z[AE_N_IN - 1]),
              "ae_normalize con %.9g no escribio el resto de z", (double)bad_vals[b]);
    }
    // Finito pero z desborda: false
    float ep3[AE_N_CH][AE_N_T];
    memcpy(ep3, ep, sizeof(ep3));
    ep3[0][0] = -FLT_MAX;
    c.std[0] = 0.25f;
    CHECK(!ae_normalize(CEPOCH(ep3), &c, z), "ae_normalize acepto una z que desborda a -inf");
    // NULL
    CHECK(!ae_normalize(NULL, &c, z) && all_nan(z, AE_N_IN), "ae_normalize(NULL, ...) debe dar false y z = NaN");
    CHECK(!ae_normalize(CEPOCH(ep), NULL, z) && all_nan(z, AE_N_IN),
          "ae_normalize(e, NULL, z) debe dar false y z = NaN");
    CHECK(!ae_normalize(CEPOCH(ep), &c, NULL), "ae_normalize(e, c, NULL) debe dar false");
}

// ---------------------------------------------------------------- (5) cuantización / descuantización

static void check_quant(int line, const float *zs, const int *expected, int n, float scale, int32_t zp)
{
    float z[AE_N_IN];
    int8_t q[AE_N_IN];
    memset(z, 0, sizeof(z));
    memcpy(z, zs, (size_t)n * sizeof(float));
    ae_quantize(z, scale, zp, q);
    for (int i = 0; i < n; i++) {
        if ((int)q[i] != expected[i]) {
            check_failed(line, "ae_quantize(%.9g, s=%.9g, zp=%d) = %d, esperado %d", (double)zs[i], (double)scale,
                         (int)zp, (int)q[i], expected[i]);
        }
    }
}

static void test_quantize(void)
{
    const ae_model_info_t *info = ae_model_info();
    int8_t q[AE_N_IN];
    int total = 0;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        ae_quantize(tv_z(k), info->in_scale, info->in_zero_point, q);
        int bad = 0;
        int first = -1;
        for (int i = 0; i < AE_N_IN; i++) {
            if (q[i] != tv_q_in(k)[i]) {
                first = (bad == 0) ? i : first;
                bad++;
            }
        }
        total += bad;
        CHECK(bad == 0, "vector %d: %d q_in distintos del golden (primero %d: %d vs %d)", k, bad, first,
              (int)q[first < 0 ? 0 : first], (int)tv_q_in(k)[first < 0 ? 0 : first]);
    }
    printf("  info: %d vectores, %d q_in distintos (bit a bit)\n", AE_TV_COUNT, total);

    // Empates exactos (escala 0.5): redondeo a par como np.rint, nunca "lejos de cero" (roundf)
    // ni truncado
    static const float tie_z[] = {0.25f, 0.75f, 1.25f, 1.75f, -0.25f, -0.75f, -1.25f, -1.75f, 0.2f, 0.3f, 0.0f, -0.0f};
    static const int tie_q0[] = {0, 2, 2, 4, 0, -2, -2, -4, 0, 1, 0, 0};
    static const int tie_qm3[] = {-3, -1, -1, 1, -3, -5, -5, -7, -3, -2, -3, -3};
    check_quant(__LINE__, tie_z, tie_q0, 12, 0.5f, 0);
    check_quant(__LINE__, tie_z, tie_qm3, 12, 0.5f, -3);

    // Saturación: clamp a [-128, 127] después de sumar zero_point (empates en el borde incluidos)
    static const float sat_z[] = {63.25f, 63.5f, 63.75f, 1000.0f, -64.0f, -64.25f, -64.75f, -1000.0f,
                                  INFINITY, -INFINITY, FLT_MAX, -FLT_MAX, NAN};
    static const int sat_q0[] = {126, 127, 127, 127, -128, -128, -128, -128, 127, -128, 127, -128, -128};
    static const int sat_q5[] = {127, 127, 127, 127, -123, -123, -125, -128, 127, -128, 127, -128, -128};
    check_quant(__LINE__, sat_z, sat_q0, 13, 0.5f, 0);
    check_quant(__LINE__, sat_z, sat_q5, 13, 0.5f, 5);
    // -66.25 / 0.5 = -132.5 -> -132 (par) + 5 = -127: con roundf sería -133 + 5 = -128
    static const float edge_z[] = {61.0f, 61.5f, -66.5f, -67.0f, -66.25f};
    static const int edge_q5[] = {127, 127, -128, -128, -127};
    check_quant(__LINE__, edge_z, edge_q5, 5, 0.5f, 5);

    ae_quantize(NULL, 0.5f, 0, q); // no debe colgarse
    ae_quantize(tv_z(0), 0.5f, 0, NULL);
}

static void test_dequantize(void)
{
    const ae_model_info_t *info = ae_model_info();
    float x[AE_N_IN];
    int total = 0;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        ae_dequantize(tv_q_out(k), info->out_scale, info->out_zero_point, x);
        int first;
        const int bad = float_mismatches(x, tv_zhat(k), AE_N_IN, &first);
        total += bad;
        CHECK(bad == 0, "vector %d: %d z_hat distintas del golden (primera %d: %.9g vs %.9g)", k, bad, first,
              (double)x[first < 0 ? 0 : first], (double)tv_zhat(k)[first < 0 ? 0 : first]);
    }
    printf("  info: %d vectores, %d z_hat distintas (bit a bit)\n", AE_TV_COUNT, total);

    // x = s * (q - zp): signo del zero point y extremos int8 (exactos con s = 0.25)
    int8_t q[AE_N_IN];
    memset(q, 0, sizeof(q));
    static const int8_t qv[6] = {-128, -1, 0, 1, 127, 42};
    static const float x_zp3[6] = {-32.75f, -1.0f, -0.75f, -0.5f, 31.0f, 9.75f};
    static const float x_zpm128[6] = {0.0f, 31.75f, 32.0f, 32.25f, 63.75f, 42.5f};
    memcpy(q, qv, sizeof(qv));
    ae_dequantize(q, 0.25f, 3, x);
    for (int i = 0; i < 6; i++) {
        CHECK(same_bits(x[i], x_zp3[i]), "ae_dequantize(%d, 0.25, 3) = %.9g, esperado %.9g", (int)qv[i], (double)x[i],
              (double)x_zp3[i]);
    }
    ae_dequantize(q, 0.25f, -128, x);
    for (int i = 0; i < 6; i++) {
        CHECK(same_bits(x[i], x_zpm128[i]), "ae_dequantize(%d, 0.25, -128) = %.9g, esperado %.9g", (int)qv[i],
              (double)x[i], (double)x_zpm128[i]);
    }
    // zero_point fuera de int8 o q NULL -> NaN (fail-safe)
    ae_dequantize(q, 0.25f, 128, x);
    CHECK(all_nan(x, AE_N_IN), "ae_dequantize con zero_point 128 debe dar NaN");
    ae_dequantize(q, 0.25f, INT32_MIN, x);
    CHECK(all_nan(x, AE_N_IN), "ae_dequantize con zero_point INT32_MIN debe dar NaN");
    ae_dequantize(NULL, 0.25f, 0, x);
    CHECK(all_nan(x, AE_N_IN), "ae_dequantize(NULL, ...) debe dar NaN");
    ae_dequantize(q, 0.25f, 0, NULL); // no debe colgarse
}

// ---------------------------------------------------------------- (6) score

static void test_score(void)
{
    float max_rel = 0.0f;
    uint32_t max_ulp = 0;
    int exact = 0;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        const float s = ae_score(tv_z(k), tv_zhat(k));
        const float e = AE_TV_SCORE[k];
        const float rel = fabsf(s - e) / fabsf(e);
        max_rel = (rel > max_rel || isnan(rel)) ? rel : max_rel;
        const uint32_t u = ulp_diff(s, e);
        max_ulp = u > max_ulp ? u : max_ulp;
        exact += same_bits(s, e) ? 1 : 0;
        CHECK(near_rel(s, e, TOL_SCORE_REL, 0.0f), "vector %d: score %.9g, golden %.9g (rel %.3g)", k, (double)s,
              (double)e, (double)rel);
    }
    printf("  info: %d scores: bit a bit %d, error relativo max %.3g (%u ulp)\n", AE_TV_COUNT, exact, (double)max_rel,
           (unsigned)max_ulp);

    // MSE dividido por 320 (todo exacto con estos valores)
    float z[AE_N_IN];
    float zh[AE_N_IN];
    for (int i = 0; i < AE_N_IN; i++) {
        z[i] = 1.0f;
        zh[i] = 0.0f;
    }
    CHECK(same_bits(ae_score(z, zh), 1.0f), "score(1, 0) = %.9g, esperado 1", (double)ae_score(z, zh));
    CHECK(same_bits(ae_score(z, z), 0.0f), "score(z, z) = %.9g, esperado 0", (double)ae_score(z, z));
    memcpy(zh, z, sizeof(zh));
    zh[AE_N_IN - 1] = -1.0f; // una sola diferencia de 2 -> 4 / 320
    CHECK(same_bits(ae_score(z, zh), 4.0f / 320.0f), "score con una diferencia de 2 = %.9g, esperado %.9g",
          (double)ae_score(z, zh), (double)(4.0f / 320.0f));
    zh[0] = NAN;
    CHECK(isnan(ae_score(z, zh)), "score con NaN debe ser NaN en bruto");
    CHECK(is_pos_inf(ae_score(NULL, z)) && is_pos_inf(ae_score(z, NULL)), "ae_score con NULL debe dar +inf");
}

// ---------------------------------------------------------------- (7) niveles (">" estricto)

// Transcripción literal del contrato, referencia para los barridos
static ae_level_t contract_level(float s, float t1, float t2, float t3)
{
    if (!isfinite(s)) {
        return AE_LEVEL_SEVERE;
    }
    if (s > t3) {
        return AE_LEVEL_SEVERE;
    }
    if (s > t2) {
        return AE_LEVEL_MODERATE;
    }
    if (s > t1) {
        return AE_LEVEL_MILD;
    }
    return AE_LEVEL_NORMAL;
}

#define CHECK_LEVEL(score, cal, expected)                                                              \
    CHECK(ae_level_from_score((score), (cal)) == (expected), "ae_level_from_score(%.9g) = %d, esperado %d", \
          (double)(score), (int)ae_level_from_score((score), (cal)), (int)(expected))

static void test_levels(void)
{
    ae_calib_t c;
    make_calib(&c, 0.0f, 1.0f, 1.0f, 2.0f, 4.0f, ae_model_info()->model_id);

    // Un score igual al umbral NO escala (">" estricto); un ulp por encima sí
    CHECK_LEVEL(0.0f, &c, AE_LEVEL_NORMAL);
    CHECK_LEVEL(-0.0f, &c, AE_LEVEL_NORMAL);
    CHECK_LEVEL(1.0f, &c, AE_LEVEL_NORMAL);
    CHECK_LEVEL(nextafterf(1.0f, INFINITY), &c, AE_LEVEL_MILD);
    CHECK_LEVEL(2.0f, &c, AE_LEVEL_MILD);
    CHECK_LEVEL(nextafterf(2.0f, INFINITY), &c, AE_LEVEL_MODERATE);
    CHECK_LEVEL(4.0f, &c, AE_LEVEL_MODERATE);
    CHECK_LEVEL(nextafterf(4.0f, INFINITY), &c, AE_LEVEL_SEVERE);
    CHECK_LEVEL(FLT_MAX, &c, AE_LEVEL_SEVERE);
    CHECK_LEVEL(NAN, &c, AE_LEVEL_SEVERE);
    CHECK_LEVEL(INFINITY, &c, AE_LEVEL_SEVERE);
    CHECK_LEVEL(-INFINITY, &c, AE_LEVEL_SEVERE);

    // Bandas vacías (t1 == t2 == t3): <= T -> 0, > T -> 3
    ae_calib_t eq;
    make_calib(&eq, 0.0f, 1.0f, 2.0f, 2.0f, 2.0f, ae_model_info()->model_id);
    CHECK_LEVEL(2.0f, &eq, AE_LEVEL_NORMAL);
    CHECK_LEVEL(nextafterf(2.0f, INFINITY), &eq, AE_LEVEL_SEVERE);

    // Umbrales inválidos o sin calibración: SEVERE siempre (también con score 0)
    ae_calib_t bad;
    const float bad_t[5][3] = {{NAN, 2.0f, 4.0f}, {1.0f, 2.0f, INFINITY}, {0.0f, 2.0f, 4.0f}, {3.0f, 2.0f, 4.0f},
                               {1.0f, 5.0f, 4.0f}};
    for (int b = 0; b < 5; b++) {
        make_calib(&bad, 0.0f, 1.0f, bad_t[b][0], bad_t[b][1], bad_t[b][2], ae_model_info()->model_id);
        CHECK(ae_level_from_score(0.0f, &bad) == AE_LEVEL_SEVERE && ae_level_from_score(1.5f, &bad) == AE_LEVEL_SEVERE,
              "umbrales invalidos (%.9g, %.9g, %.9g) no dan SEVERE", (double)bad_t[b][0], (double)bad_t[b][1],
              (double)bad_t[b][2]);
    }
    CHECK(ae_level_from_score(0.0f, NULL) == AE_LEVEL_SEVERE, "ae_level_from_score(0, NULL) debe ser SEVERE");

    // Barridos con la calibración por defecto: +-4 ulp alrededor de cada umbral y escala
    // logarítmica de 1e-3 T1 a 1e3 T3
    ae_calib_t def;
    ae_calib_default(&def);
    const float ts[3] = {def.t1, def.t2, def.t3};
    for (int k = 0; k < 3; k++) {
        float s = ts[k];
        for (int j = 0; j < 4; j++) {
            s = nextafterf(s, 0.0f);
        }
        for (int j = 0; j < 9; j++) {
            CHECK_LEVEL(s, &def, contract_level(s, def.t1, def.t2, def.t3));
            s = nextafterf(s, INFINITY);
        }
        CHECK(ae_level_from_score(ts[k], &def) < (ae_level_t)(k + 1),
              "score == T%d (%.9g) ya da nivel %d (debe ser '>' estricto)", k + 1, (double)ts[k], k + 1);
    }
    for (int j = 0; j <= 600; j++) {
        const float s = def.t1 * powf(10.0f, -3.0f + 0.01f * (float)j);
        CHECK_LEVEL(s, &def, contract_level(s, def.t1, def.t2, def.t3));
    }
}

// ---------------------------------------------------------------- (8) inferencia completa vs goldens

static void test_infer_goldens(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    int q_out_vec_bad = 0;
    int q_out_lsb = 0;
    int q_out_maxdiff = 0;
    int score_exact = 0;
    int level_checked = 0;
    float max_rel = 0.0f;

    for (int k = 0; k < AE_TV_COUNT; k++) {
        const char *why = "";
#ifdef AE_TEST_FAKE_RUNNER
        const int miss0 = g_fake.n_miss;
#endif
        const ae_result_t r = ae_infer(tv_epoch(k), &def, &g_ws);
#ifdef AE_TEST_FAKE_RUNNER
        if (g_fake.n_miss != miss0) {
            why = " (runner falso: q_in sin golden exacto)";
        }
#endif
        CHECK(r.valid, "vector %d: ae_infer devolvio valid=false con una epoca valida%s", k, why);
        if (!r.valid) {
            continue;
        }
        int first;
        int bad = float_mismatches(g_ws.z, tv_z(k), AE_N_IN, &first);
        CHECK(bad == 0, "vector %d: %d z distintas del golden (primera %d)", k, bad, first);
        bad = 0;
        for (int i = 0; i < AE_N_IN; i++) {
            bad += g_ws.q_in[i] != tv_q_in(k)[i];
        }
        CHECK(bad == 0, "vector %d: %d q_in distintos del golden", k, bad);

        // q_out del runner (TFLM en el nivel B) frente al golden: bit a bit
        int vbad = 0;
        for (int i = 0; i < AE_N_IN; i++) {
            const int d = (int)g_ws.q_out[i] - (int)tv_q_out(k)[i];
            if (d != 0) {
                if (vbad == 0) {
                    check_failed(__LINE__, "vector %d: q_out[%d] = %d, golden %d (diferencia %d LSB)", k, i,
                                 (int)g_ws.q_out[i], (int)tv_q_out(k)[i], d);
                }
                vbad++;
                q_out_lsb++;
                q_out_maxdiff = (d > q_out_maxdiff) ? d : (-d > q_out_maxdiff ? -d : q_out_maxdiff);
            }
        }
        q_out_vec_bad += vbad > 0;

        bad = float_mismatches(g_ws.z_hat, tv_zhat(k), AE_N_IN, &first);
        CHECK(vbad > 0 || bad == 0, "vector %d: %d z_hat distintas del golden con q_out exacto", k, bad);

        const float e = AE_TV_SCORE[k];
        const float rel = fabsf(r.score - e) / fabsf(e);
        max_rel = (rel > max_rel || isnan(rel)) ? rel : max_rel;
        score_exact += same_bits(r.score, e) ? 1 : 0;
        CHECK(near_rel(r.score, e, TOL_SCORE_REL, 0.0f), "vector %d: score %.9g, golden %.9g (rel %.3g)", k,
              (double)r.score, (double)e, (double)rel);
        CHECK(same_bits(r.score, ae_score(g_ws.z, g_ws.z_hat)), "vector %d: score de ae_infer != ae_score(ws)", k);
        CHECK(r.level == ae_level_from_score(r.score, &def), "vector %d: nivel incoherente con su score", k);
        if (AE_TV_NEAR_THRESHOLD[k] == 0) {
            // Cerca de un umbral el redondeo puede cambiar el nivel: solo se exige lejos de ellos
            level_checked++;
            CHECK((int)r.level == (int)AE_TV_LEVEL[k], "vector %d: nivel %d, golden %d (score %.9g, golden %.9g)", k,
                  (int)r.level, (int)AE_TV_LEVEL[k], (double)r.score, (double)e);
        }
    }
    printf("  info: %d vectores: q_out bit a bit en %d (%d LSB distintos, dif max %d); score bit a bit en %d, "
           "error relativo max %.3g; nivel exigido en %d\n",
           AE_TV_COUNT, AE_TV_COUNT - q_out_vec_bad, q_out_lsb, q_out_maxdiff, score_exact, (double)max_rel,
           level_checked);
    if (q_out_lsb > 0) {
        printf("  pista: diferencias sueltas de 1 LSB en q_out suelen ser goldens de tf.lite BUILTIN_REF, cuyo "
               "FULLY_CONNECTED redondea distinto que TFLM; los goldens deben replicar la aritmetica de TFLM "
               "(regenerar con export_to_c.py sin --golden-runtime builtin_ref)\n");
    }
}

// Cada banda de nivel no vacía con la calibración por defecto tiene >= 1 vector dorado
static void test_golden_coverage(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    int count[4] = {0, 0, 0, 0};
    for (int k = 0; k < AE_TV_COUNT; k++) {
        const int lv = (int)AE_TV_LEVEL[k];
        CHECK(lv >= 0 && lv <= 3, "vector %d: AE_TV_LEVEL = %d fuera de 0..3", k, lv);
        if (lv >= 0 && lv <= 3) {
            count[lv]++;
        }
    }
    const bool nonempty[4] = {true, def.t1 < def.t2, def.t2 < def.t3, true};
    for (int lv = 0; lv <= 3; lv++) {
        CHECK(!nonempty[lv] || count[lv] >= 1,
              "ningun vector dorado de nivel %d: regenerar con el exportador (cobertura por escala + biseccion)", lv);
    }
    printf("  info: vectores por nivel: N0=%d N1=%d N2=%d N3=%d\n", count[0], count[1], count[2], count[3]);
}

// ---------------------------------------------------------------- (9) fail-safe

static void test_fail_safe(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    char what[128];
#ifdef AE_TEST_FAKE_RUNNER
    const int inv0 = g_fake.n_invoke;
#endif

    // NaN / +Inf / -Inf en la primera, una intermedia y la última muestra: la red no se ejecuta
    const float bad[3] = {NAN, INFINITY, -INFINITY};
    const char *bad_name[3] = {"NaN", "+Inf", "-Inf"};
    const int pos_c[3] = {0, 3, AE_N_CH - 1};
    const int pos_t[3] = {0, 17, AE_N_T - 1};
    for (int b = 0; b < 3; b++) {
        for (int p = 0; p < 3; p++) {
            float ep[AE_N_CH][AE_N_T];
            memcpy(ep, tv_epoch(0), sizeof(ep));
            ep[pos_c[p]][pos_t[p]] = bad[b];
            snprintf(what, sizeof(what), "%s en epoch[%d][%d]", bad_name[b], pos_c[p], pos_t[p]);
            check_invalid_result(__LINE__, ae_infer(CEPOCH(ep), &def, &g_ws), what);
        }
    }
#ifdef AE_TEST_FAKE_RUNNER
    CHECK(g_fake.n_invoke == inv0, "el runner se invoco con epocas no finitas");
#endif

    // Finito pero enorme: z desborda (normalize) o, con z finita ~1e29, la red se ejecuta, q_in
    // satura y d^2 desborda el score a +inf: inválido siempre. En el nivel A el runner falso
    // devuelve q_out = q_in ("eco") para que la cadena llegue de verdad al score.
#ifdef AE_TEST_FAKE_RUNNER
    g_fake.echo = true;
    const int inv_huge = g_fake.n_invoke;
#endif
    const float huge[3] = {1e30f, -1e30f, FLT_MAX};
    for (int h = 0; h < 3; h++) {
        for (int all = 0; all < 2; all++) {
            float ep[AE_N_CH][AE_N_T];
            memcpy(ep, tv_epoch(0), sizeof(ep));
            if (all) {
                for (int ch = 0; ch < AE_N_CH; ch++) {
                    for (int t = 0; t < AE_N_T; t++) {
                        ep[ch][t] = huge[h];
                    }
                }
            } else {
                ep[2][9] = huge[h];
            }
            snprintf(what, sizeof(what), "epoca con %.9g (%s)", (double)huge[h], all ? "todas las muestras" : "una");
            check_invalid_result(__LINE__, ae_infer(CEPOCH(ep), &def, &g_ws), what);
        }
    }
#ifdef AE_TEST_FAKE_RUNNER
    g_fake.echo = false;
    CHECK(g_fake.n_invoke > inv_huge, "ningun caso enorme llego al runner: el desbordamiento del score no se probo");
#endif
    // Desbordamiento solo en el score (z finita): ae_score da +inf en bruto
    {
        float z[AE_N_IN];
        float zh[AE_N_IN];
        for (int i = 0; i < AE_N_IN; i++) {
            z[i] = 1e20f;
            zh[i] = 0.0f;
        }
        CHECK(is_pos_inf(ae_score(z, zh)), "ae_score con d = 1e20 debe desbordar a +inf");
    }

    // Toda calibración inválida: ae_calib_check false y ae_infer inválido sin ejecutar la red
    typedef struct {
        const char *name;
        int field; // 0 mean, 1 std, 2 t1, 3 t2, 4 t3, 5 model_id
        int index;
        float value;
        const char *id;
    } bad_cal_t;
    char id_last[AE_MODEL_ID_LEN + 1];
    char id_first[AE_MODEL_ID_LEN + 1];
    char id_upper[AE_MODEL_ID_LEN + 1];
    memcpy(id_last, def.model_id, sizeof(id_last));
    memcpy(id_first, def.model_id, sizeof(id_first));
    memcpy(id_upper, def.model_id, sizeof(id_upper));
    id_last[AE_MODEL_ID_LEN - 1] = (char)(id_last[AE_MODEL_ID_LEN - 1] == '0' ? '1' : '0');
    id_first[0] = (char)(id_first[0] == 'a' ? 'b' : 'a');
    for (int i = 0; i < AE_MODEL_ID_LEN; i++) {
        id_upper[i] = (id_upper[i] >= 'a' && id_upper[i] <= 'f') ? (char)(id_upper[i] - 'a' + 'A') : id_upper[i];
    }
    bool has_alpha = false;
    for (int i = 0; i < AE_MODEL_ID_LEN; i++) {
        has_alpha = has_alpha || (def.model_id[i] >= 'a' && def.model_id[i] <= 'f');
    }
    const bad_cal_t cases[] = {
        {"mean[0] NaN", 0, 0, NAN, NULL},        {"mean[7] +Inf", 0, 7, INFINITY, NULL},
        {"mean[3] -Inf", 0, 3, -INFINITY, NULL}, {"std[0] = 0", 1, 0, 0.0f, NULL},
        {"std[7] = -0", 1, 7, -0.0f, NULL},      {"std[2] < 0", 1, 2, -1.0f, NULL},
        {"std[5] NaN", 1, 5, NAN, NULL},         {"std[1] +Inf", 1, 1, INFINITY, NULL},
        {"std[6] -Inf", 1, 6, -INFINITY, NULL},  {"t1 = 0", 2, 0, 0.0f, NULL},
        {"t1 < 0", 2, 0, -1.0f, NULL},           {"t1 NaN", 2, 0, NAN, NULL},
        {"t1 > t2", 2, 0, 1e30f, NULL},          {"t2 > t3", 3, 0, 1e30f, NULL},
        {"t2 NaN", 3, 0, NAN, NULL},             {"t3 +Inf", 4, 0, INFINITY, NULL},
        {"t3 NaN", 4, 0, NAN, NULL},             {"t3 < t2", 4, 0, 0.0f, NULL},
        {"model_id ultimo caracter", 5, 0, 0.0f, id_last},
        {"model_id primer caracter", 5, 0, 0.0f, id_first},
        {"model_id en mayusculas", 5, 0, 0.0f, has_alpha ? id_upper : id_first},
        {"model_id vacio", 5, 0, 0.0f, ""},
        {"model_id truncado (15)", 5, 0, 0.0f, NULL},
        {"model_id sin terminador", 5, 0, 0.0f, NULL},
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
#ifdef AE_TEST_FAKE_RUNNER
    const int inv1 = g_fake.n_invoke;
#endif
    for (int i = 0; i < n_cases; i++) {
        ae_calib_t c = def;
        switch (cases[i].field) {
        case 0:
            c.mean[cases[i].index] = cases[i].value;
            break;
        case 1:
            c.std[cases[i].index] = cases[i].value;
            break;
        case 2:
            c.t1 = cases[i].value;
            break;
        case 3:
            c.t2 = cases[i].value;
            break;
        case 4:
            c.t3 = cases[i].value;
            break;
        default:
            if (cases[i].id != NULL) {
                set_model_id(&c, cases[i].id);
            } else if (strcmp(cases[i].name, "model_id truncado (15)") == 0) {
                c.model_id[AE_MODEL_ID_LEN - 1] = '\0';
            } else {
                c.model_id[AE_MODEL_ID_LEN] = 'x'; // 17 caracteres sin '\0'
            }
            break;
        }
        CHECK(!ae_calib_check(&c), "calibracion invalida (%s) aceptada por ae_calib_check", cases[i].name);
        snprintf(what, sizeof(what), "calibracion invalida (%s)", cases[i].name);
        check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &c, &g_ws), what);
    }
#ifdef AE_TEST_FAKE_RUNNER
    CHECK(g_fake.n_invoke == inv1, "el runner se invoco con una calibracion invalida");
#endif
    CHECK(ae_calib_check(&def) && ae_infer(tv_epoch(0), &def, &g_ws).valid,
          "la calibracion por defecto dejo de funcionar tras los casos invalidos");

    // Punteros NULL
    check_invalid_result(__LINE__, ae_infer(NULL, &def, &g_ws), "epoch NULL");
    check_invalid_result(__LINE__, ae_infer(tv_epoch(0), NULL, &g_ws), "calibracion NULL");
    check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &def, NULL), "workspace NULL");
    CHECK(!ae_calib_check(NULL), "ae_calib_check(NULL) = true");
    ae_calib_default(NULL); // no debe colgarse

    // Ventana con NaN -> época con NaN -> inválido (cadena completa ventana -> nivel)
    static float win[AE_N_CH][AE_WIN_SAMPLES];
    float ep[AE_N_CH][AE_N_T];
    memcpy(win, tv_win(0), sizeof(win));
    win[6][123] = NAN;
    ae_preprocess(CWIN(win), ep);
    check_invalid_result(__LINE__, ae_infer(CEPOCH(ep), &def, &g_ws), "ventana con NaN");
}

#ifdef AE_TEST_FAKE_RUNNER
// Fallos del runner (solo nivel A): init, invoke y metadatos corruptos que ae_init rechaza
static void test_runner_faults(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    const ae_result_t ref = ae_infer(tv_epoch(0), &def, &g_ws);
    CHECK(ref.valid, "la inferencia de referencia no es valida");

    // Invoke falla -> inválido; al recuperarse, mismo resultado
    g_fake.fail_invoke = true;
    check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &def, &g_ws), "runner: invoke falla");
    g_fake.fail_invoke = false;
    CHECK(same_result(ae_infer(tv_epoch(0), &def, &g_ws), ref), "resultado distinto tras un fallo de invoke");

    // Init falla -> motor sin inicializar: todo fail-safe hasta el siguiente ae_init correcto
    g_fake.fail_init = true;
    CHECK(!ae_init() && !ae_is_ready(), "ae_init devolvio true con el runner fallando");
    CHECK(ae_model_info()->model_id[0] == '\0', "ae_model_info conserva el modelo tras un ae_init fallido");
    CHECK(!ae_calib_check(&def), "ae_calib_check acepta calibraciones sin modelo cargado");
    check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &def, &g_ws), "runner: init falla");
    ae_calib_t c;
    ae_calib_default(&c);
    CHECK(!ae_calib_check(&c) && isnan(c.t1), "ae_calib_default sin modelo debe ser invalida");
    g_fake.fail_init = false;

    // Metadatos corruptos: ae_init los rechaza todos
    for (int k = 1; k < FAKE_N_CORRUPT; k++) {
        g_fake.corrupt = k;
        const bool ok = ae_init();
        CHECK(!ok && !ae_is_ready(), "ae_init acepto metadatos corruptos: %s", FAKE_CORRUPT_NAME[k]);
        if (!ok) { // si los aceptó, no seguir usándolos (p.ej. model_id NULL)
            check_invalid_result(__LINE__, ae_infer(tv_epoch(0), &def, &g_ws), FAKE_CORRUPT_NAME[k]);
        }
    }
    g_fake.corrupt = FAKE_OK;

    // Recuperación
    CHECK(ae_init() && ae_is_ready(), "ae_init no se recupero tras los fallos inyectados");
    CHECK(same_result(ae_infer(tv_epoch(0), &def, &g_ws), ref), "resultado distinto tras reinicializar");
    CHECK(!ae_runner_init(NULL), "ae_runner_init(NULL) debe fallar");
}
#else
// Runner real (niveles B y C): argumentos inválidos y llamada directa con un q_in dorado
static void test_runner_real(void)
{
    int8_t out[AE_N_IN];
    CHECK(!ae_runner_invoke(NULL, out) && !ae_runner_invoke(tv_q_in(0), NULL), "ae_runner_invoke con NULL debe fallar");
    CHECK(!ae_runner_init(NULL), "ae_runner_init(NULL) debe fallar");
    for (int k = 0; k < AE_TV_COUNT; k++) {
        memset(out, 0x55, sizeof(out));
        const bool ok = ae_runner_invoke(tv_q_in(k), out);
        CHECK(ok && memcmp(out, tv_q_out(k), AE_N_IN) == 0, "vector %d: ae_runner_invoke(q_in dorado) != q_out dorado",
              k);
    }
}
#endif

// ---------------------------------------------------------------- (10) calibración

// Referencias en double (test, no motor): media / desviación poblacional en dos pasadas y
// percentil "linear" de numpy (índice (n - 1) * q y _lerp con la rama t >= 0.5)
static void ref_mean_std(const float *const *epochs, int n_ep, int ch, double *mean, double *std)
{
    double s = 0.0;
    for (int e = 0; e < n_ep; e++) {
        for (int t = 0; t < AE_N_T; t++) {
            s += (double)epochs[e][ch * AE_N_T + t];
        }
    }
    const double n = (double)n_ep * AE_N_T;
    const double m = s / n;
    double v = 0.0;
    for (int e = 0; e < n_ep; e++) {
        for (int t = 0; t < AE_N_T; t++) {
            const double d = (double)epochs[e][ch * AE_N_T + t] - m;
            v += d * d;
        }
    }
    *mean = m;
    *std = sqrt(v / n);
}

static double ref_percentile(const float *sorted, int n, double pct)
{
    const double pos = (double)(n - 1) * (pct / 100.0);
    const double lo_d = floor(pos);
    int lo = (int)lo_d;
    lo = lo > n - 1 ? n - 1 : lo;
    const int hi = lo + 1 < n ? lo + 1 : n - 1;
    const double t = pos - lo_d;
    const double a = (double)sorted[lo];
    const double b = (double)sorted[hi];
    const double diff = b - a;
    return t >= 0.5 ? b - diff * (1.0 - t) : a + diff * t;
}

static void sort_ref(float *v, int n) // inserción (independiente del heapsort del motor)
{
    for (int i = 1; i < n; i++) {
        const float x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }
}

static void test_calibration_goldens(void)
{
    // mean/std de las épocas doradas (golden: numpy float64 -> float32)
    ae_norm_acc_t acc;
    ae_norm_acc_init(&acc);
    for (int e = 0; e < AE_TV_CAL_N_EPOCHS; e++) {
        ae_norm_acc_add(&acc, tv_cal_epoch(e));
    }
    ae_calib_t c;
    ae_calib_default(&c);
    const ae_calib_t before = c;
    CHECK(ae_norm_acc_finish(&acc, &c), "ae_norm_acc_finish fallo con las epocas doradas");
    CHECK(acc.n == (uint32_t)(AE_TV_CAL_N_EPOCHS * AE_N_T), "acc.n = %u, esperado %d", (unsigned)acc.n,
          AE_TV_CAL_N_EPOCHS * AE_N_T);
    float max_m = 0.0f;
    float max_s = 0.0f;
    for (int ch = 0; ch < AE_N_CH; ch++) {
        const float em = fabsf(c.mean[ch] - AE_TV_CAL_MEAN[ch]) / AE_TV_CAL_STD[ch];
        const float es = fabsf(c.std[ch] - AE_TV_CAL_STD[ch]) / AE_TV_CAL_STD[ch];
        max_m = em > max_m ? em : max_m;
        max_s = es > max_s ? es : max_s;
        CHECK(em <= TOL_STD_REL, "canal %d: mean %.9g, golden %.9g", ch, (double)c.mean[ch],
              (double)AE_TV_CAL_MEAN[ch]);
        CHECK(es <= TOL_STD_REL, "canal %d: std %.9g, golden %.9g (rel %.3g; ddof 0)", ch, (double)c.std[ch],
              (double)AE_TV_CAL_STD[ch], (double)es);
    }
    CHECK(same_bits(c.t1, before.t1) && same_bits(c.t3, before.t3) && strcmp(c.model_id, before.model_id) == 0,
          "ae_norm_acc_finish toco umbrales o model_id");

    // Umbrales de los scores dorados; además ordena en el sitio
    float scores[AE_TV_CAL_N_SCORES];
    float sorted[AE_TV_CAL_N_SCORES];
    memcpy(scores, AE_TV_CAL_SCORES, sizeof(scores));
    memcpy(sorted, AE_TV_CAL_SCORES, sizeof(sorted));
    sort_ref(sorted, AE_TV_CAL_N_SCORES);
    CHECK(ae_calib_thresholds_from_scores(&c, scores, AE_TV_CAL_N_SCORES), "umbrales doradas: fallo");
    CHECK(memcmp(scores, sorted, sizeof(scores)) == 0,
          "ae_calib_thresholds_from_scores no ordeno los scores en el sitio");
    const float t[3] = {c.t1, c.t2, c.t3};
    float max_t = 0.0f;
    for (int k = 0; k < 3; k++) {
        const float et = fabsf(t[k] - AE_TV_CAL_T[k]) / fabsf(AE_TV_CAL_T[k]);
        max_t = et > max_t ? et : max_t;
        CHECK(et <= TOL_PCT_REL, "T%d = %.9g, golden %.9g (rel %.3g)", k + 1, (double)t[k], (double)AE_TV_CAL_T[k],
              (double)et);
    }
    CHECK(ae_calib_check(&c), "la calibracion dorada completa no pasa ae_calib_check");
    printf("  info: %d epocas: error max mean %.3g std, std %.3g rel; %d scores: umbrales %.3g rel\n",
           AE_TV_CAL_N_EPOCHS, (double)max_m, (double)max_s, AE_TV_CAL_N_SCORES, (double)max_t);
}

static void test_calibration_reference(void)
{
    // Welford (motor, float32) frente a dos pasadas en double con 1, 3 y 20 épocas aleatorias
    // de media y escala distintas por canal
    static float eps[20][AE_N_CH][AE_N_T];
    const float *ptrs[20];
    g_rng = 2024u;
    for (int e = 0; e < 20; e++) {
        for (int ch = 0; ch < AE_N_CH; ch++) {
            for (int t = 0; t < AE_N_T; t++) {
                eps[e][ch][t] = 0.7f * (float)ch - 2.0f + (3.0f + 2.0f * (float)ch) * rnd_normal();
            }
        }
        ptrs[e] = &eps[e][0][0];
    }
    const int n_eps[3] = {1, 3, 20};
    for (int r = 0; r < 3; r++) {
        ae_norm_acc_t acc;
        ae_norm_acc_init(&acc);
        for (int e = 0; e < n_eps[r]; e++) {
            ae_norm_acc_add(&acc, CEPOCH(eps[e]));
        }
        ae_calib_t c;
        ae_calib_default(&c);
        CHECK(ae_norm_acc_finish(&acc, &c), "%d epocas: ae_norm_acc_finish fallo", n_eps[r]);
        for (int ch = 0; ch < AE_N_CH; ch++) {
            double m;
            double s;
            ref_mean_std(ptrs, n_eps[r], ch, &m, &s);
            CHECK(fabs((double)c.mean[ch] - m) <= (double)TOL_STD_REL * s, "%d epocas, canal %d: mean %.9g, ref %.9g",
                  n_eps[r], ch, (double)c.mean[ch], m);
            CHECK(fabs((double)c.std[ch] - s) <= (double)TOL_STD_REL * s,
                  "%d epocas, canal %d: std %.9g, ref %.9g (ddof 0)", n_eps[r], ch, (double)c.std[ch], s);
        }
    }

    // Errores del acumulador: sin muestras, canal plano, NaN, NULL -> false y mean/std NaN
    ae_norm_acc_t acc;
    ae_calib_t c;
    ae_calib_default(&c);
    ae_norm_acc_init(&acc);
    CHECK(!ae_norm_acc_finish(&acc, &c) && isnan(c.mean[0]) && isnan(c.std[7]), "acumulador vacio: debe fallar");
    float flat[AE_N_CH][AE_N_T];
    memcpy(flat, eps[0], sizeof(flat));
    for (int t = 0; t < AE_N_T; t++) {
        flat[4][t] = 3.25f; // canal plano: std = 0
    }
    ae_norm_acc_init(&acc);
    ae_norm_acc_add(&acc, CEPOCH(flat));
    ae_calib_default(&c);
    CHECK(!ae_norm_acc_finish(&acc, &c) && isnan(c.std[0]), "canal plano (std 0): debe fallar");
    memcpy(flat, eps[0], sizeof(flat));
    flat[1][5] = NAN;
    ae_norm_acc_init(&acc);
    ae_norm_acc_add(&acc, CEPOCH(eps[1]));
    ae_norm_acc_add(&acc, CEPOCH(flat));
    ae_norm_acc_add(&acc, CEPOCH(eps[2]));
    ae_calib_default(&c);
    CHECK(!ae_norm_acc_finish(&acc, &c), "epoca con NaN: debe fallar");
    ae_norm_acc_init(&acc);
    ae_norm_acc_add(&acc, NULL);
    ae_norm_acc_add(&acc, CEPOCH(eps[1]));
    ae_calib_default(&c);
    CHECK(!ae_norm_acc_finish(&acc, &c), "ae_norm_acc_add(NULL) debe envenenar el acumulador");
    CHECK(!ae_norm_acc_finish(NULL, &c) && !ae_norm_acc_finish(&acc, NULL), "ae_norm_acc_finish con NULL");
    ae_norm_acc_init(NULL); // no deben colgarse
    ae_norm_acc_add(NULL, CEPOCH(eps[0]));

    // Percentiles frente a numpy "linear" (double) con n = 20, 21, 60, 100, 257
    static float sc[257];
    const int ns[5] = {20, 21, 60, 100, 257};
    const float pcts[9] = {0.0f, 1.0f, 50.0f, 90.0f, 97.0f, 99.0f, 99.5f, 99.9f, 100.0f};
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < ns[r]; i++) {
            sc[i] = expf(0.5f * rnd_normal());
        }
        sort_ref(sc, ns[r]);
        for (int p = 0; p < 9; p++) {
            const float v = ae_percentile_sorted(sc, ns[r], pcts[p]);
            const double ref = ref_percentile(sc, ns[r], (double)pcts[p]);
            CHECK(fabs((double)v - ref) <= (double)TOL_PCT_REL * fabs(ref), "n=%d p%g: %.9g, numpy %.9g", ns[r],
                  (double)pcts[p], (double)v, ref);
        }
    }
    // Colas pesadas y n grande: la parte fraccionaria del índice debe conservar precisión
    // relativa (calcular pos en float32 y restarle lo daba ~5e-5 relativo en T)
    static float heavy[1000];
    for (int i = 0; i < 1000; i++) {
        heavy[i] = expf(2.5f * rnd_normal());
    }
    sort_ref(heavy, 1000);
    const float hp[4] = {90.0f, 97.0f, 99.0f, 99.5f};
    for (int p = 0; p < 4; p++) {
        const float v = ae_percentile_sorted(heavy, 1000, hp[p]);
        const double ref = ref_percentile(heavy, 1000, (double)hp[p]);
        CHECK(fabs((double)v - ref) <= (double)TOL_PCT_REL * fabs(ref), "colas pesadas n=1000 p%g: %.9g, numpy %.9g",
              (double)hp[p], (double)v, ref);
    }

    // Casos exactos y argumentos inválidos
    static const float s5[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CHECK(same_bits(ae_percentile_sorted(s5, 5, 0.0f), 1.0f) && same_bits(ae_percentile_sorted(s5, 5, 25.0f), 2.0f) &&
              same_bits(ae_percentile_sorted(s5, 5, 50.0f), 3.0f) &&
              same_bits(ae_percentile_sorted(s5, 5, 62.5f), 3.5f) &&
              same_bits(ae_percentile_sorted(s5, 5, 100.0f), 5.0f),
          "percentiles exactos de {1..5} incorrectos");
    CHECK(near_rel(ae_percentile_sorted(s5, 5, 90.0f), 4.6f, 1e-6f, 0.0f), "p90 de {1..5} = %.9g, esperado 4.6",
          (double)ae_percentile_sorted(s5, 5, 90.0f));
    CHECK(same_bits(ae_percentile_sorted(s5, 1, 73.0f), 1.0f), "percentil con n = 1 debe ser s[0]");
    CHECK(isnan(ae_percentile_sorted(s5, 0, 50.0f)) && isnan(ae_percentile_sorted(s5, -3, 50.0f)) &&
              isnan(ae_percentile_sorted(NULL, 5, 50.0f)) && isnan(ae_percentile_sorted(s5, 5, -1.0f)) &&
              isnan(ae_percentile_sorted(s5, 5, 100.5f)) && isnan(ae_percentile_sorted(s5, 5, NAN)),
          "ae_percentile_sorted con argumentos invalidos debe dar NaN");

    // ae_calib_thresholds_from_scores: mínimo de scores, no finitos, degenerados y NULL
    float s[64];
    for (int i = 0; i < 64; i++) {
        s[i] = 1.0f + 0.01f * (float)((i * 37) % 64);
    }
    ae_calib_default(&c);
    CHECK(!ae_calib_thresholds_from_scores(&c, s, AE_CALIB_MIN_SCORES - 1) && isnan(c.t1) && isnan(c.t2) &&
              isnan(c.t3) && !ae_calib_check(&c),
          "%d scores (< minimo) deben fallar y dejar umbrales NaN", AE_CALIB_MIN_SCORES - 1);
    ae_calib_default(&c);
    CHECK(ae_calib_thresholds_from_scores(&c, s, AE_CALIB_MIN_SCORES) && ae_calib_check(&c),
          "%d scores (minimo) deben bastar", AE_CALIB_MIN_SCORES);
    s[7] = NAN;
    CHECK(!ae_calib_thresholds_from_scores(&c, s, 64) && isnan(c.t1), "un score NaN debe hacer fallar los umbrales");
    s[7] = INFINITY;
    CHECK(!ae_calib_thresholds_from_scores(&c, s, 64), "un score +Inf debe hacer fallar los umbrales");
    for (int i = 0; i < 64; i++) {
        s[i] = 0.0f;
    }
    CHECK(!ae_calib_thresholds_from_scores(&c, s, 64), "scores todos 0 (T1 = 0) deben fallar");
    CHECK(!ae_calib_thresholds_from_scores(NULL, s, 64) && !ae_calib_thresholds_from_scores(&c, NULL, 64),
          "ae_calib_thresholds_from_scores con NULL debe fallar");
}

// Flujo completo de calibración (como en el firmware): defecto -> mean/std -> scores -> umbrales
static void test_calibration_flow(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    ae_calib_t cal = def;
    ae_norm_acc_t acc;
    ae_norm_acc_init(&acc);
    for (int e = 0; e < AE_TV_CAL_N_EPOCHS; e++) {
        ae_norm_acc_add(&acc, tv_cal_epoch(e));
    }
    CHECK(ae_norm_acc_finish(&acc, &cal) && ae_calib_check(&cal), "mean/std nuevas: calibracion invalida");

#ifdef AE_TEST_FAKE_RUNNER
    // El runner falso solo conoce los q_in dorados: los scores salen de la calibración por defecto
    const ae_calib_t *score_cal = &def;
#else
    const ae_calib_t *score_cal = &cal;
#endif
    float scores[AE_TV_COUNT + AE_TV_CAL_N_EPOCHS + AE_CALIB_MIN_SCORES];
    int n = 0;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        const ae_result_t r = ae_infer(tv_epoch(k), score_cal, &g_ws);
        CHECK(r.valid, "flujo: vector %d invalido", k);
        if (r.valid) {
            scores[n++] = r.score;
        }
    }
#ifndef AE_TEST_FAKE_RUNNER
    for (int e = 0; e < AE_TV_CAL_N_EPOCHS; e++) {
        const ae_result_t r = ae_infer(tv_cal_epoch(e), score_cal, &g_ws);
        CHECK(r.valid, "flujo: epoca de calibracion %d invalida", e);
        if (r.valid) {
            scores[n++] = r.score;
        }
    }
#endif
    for (int i = 0; n > 0 && n < AE_CALIB_MIN_SCORES; i++) {
        scores[n++] = scores[i]; // pocos vectores dorados: repetir (n >= 20)
    }
    float sorted[AE_TV_COUNT + AE_TV_CAL_N_EPOCHS + AE_CALIB_MIN_SCORES];
    memcpy(sorted, scores, (size_t)n * sizeof(float));
    sort_ref(sorted, n);
    CHECK(ae_calib_thresholds_from_scores(&cal, scores, n), "flujo: umbrales fallaron con %d scores", n);
    const ae_model_info_t *info = ae_model_info();
    CHECK(same_bits(cal.t1, ae_percentile_sorted(sorted, n, info->calib_pct[0])) &&
              same_bits(cal.t2, ae_percentile_sorted(sorted, n, info->calib_pct[1])) &&
              same_bits(cal.t3, ae_percentile_sorted(sorted, n, info->calib_pct[2])),
          "flujo: umbrales distintos de los percentiles del modelo");
    CHECK(ae_calib_check(&cal) && strcmp(cal.model_id, def.model_id) == 0, "flujo: la calibracion final no es valida");
    const ae_result_t r = ae_infer(tv_epoch(0), score_cal, &g_ws);
    CHECK(r.valid, "flujo: inferencia con la calibracion final invalida");
    printf("  info: flujo con %d scores: T = %.6g / %.6g / %.6g\n", n, (double)cal.t1, (double)cal.t2, (double)cal.t3);
}

// ---------------------------------------------------------------- (11) determinismo y workspace

static bool same_ws(const ae_workspace_t *a, const ae_workspace_t *b)
{
    return memcmp(a->z, b->z, sizeof(a->z)) == 0 && memcmp(a->q_in, b->q_in, sizeof(a->q_in)) == 0 &&
           memcmp(a->q_out, b->q_out, sizeof(a->q_out)) == 0 && memcmp(a->z_hat, b->z_hat, sizeof(a->z_hat)) == 0;
}

static void test_determinism(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    const ae_calib_t def_copy = def;
    for (int k = 0; k < AE_TV_COUNT; k++) {
        float ep[AE_N_CH][AE_N_T];
        memcpy(ep, tv_epoch(k), sizeof(ep));
        const ae_result_t a = ae_infer(CEPOCH(ep), &def, &g_ws);
        (void)ae_infer(tv_epoch((k + 1) % AE_TV_COUNT), &def, &g_ws2); // otra entrada entre medias
        const ae_result_t b = ae_infer(CEPOCH(ep), &def, &g_ws2);
        CHECK(same_result(a, b) && same_ws(&g_ws, &g_ws2), "vector %d: resultados distintos en llamadas repetidas", k);
        CHECK(memcmp(ep, tv_epoch(k), sizeof(ep)) == 0, "vector %d: la inferencia modifico la epoca", k);
    }
    CHECK(memcmp(&def, &def_copy, sizeof(def)) == 0, "la inferencia modifico la calibracion");
}

static void test_workspace_reuse(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    static ae_result_t ref[AE_TV_COUNT];
    static ae_workspace_t ref_ws[AE_TV_COUNT];
    for (int k = 0; k < AE_TV_COUNT; k++) {
        memset(&g_ws, 0, sizeof(g_ws));
        ref[k] = ae_infer(tv_epoch(k), &def, &g_ws);
        ref_ws[k] = g_ws;
    }
    // Un solo workspace reutilizado en orden inverso, con basura previa (0xFF: NaN y -1) y
    // tras una llamada fallida que deja NaN dentro
    memset(&g_ws, 0xFF, sizeof(g_ws));
    for (int k = AE_TV_COUNT - 1; k >= 0; k--) {
        float ep[AE_N_CH][AE_N_T];
        memcpy(ep, tv_epoch(k), sizeof(ep));
        ep[1][1] = NAN;
        check_invalid_result(__LINE__, ae_infer(CEPOCH(ep), &def, &g_ws), "epoca con NaN entre medias");
        const ae_result_t r = ae_infer(tv_epoch(k), &def, &g_ws);
        CHECK(same_result(r, ref[k]) && same_ws(&g_ws, &ref_ws[k]),
              "vector %d: resultado distinto con el workspace reutilizado", k);
    }
}

// ---------------------------------------------------------------- (12) benchmark

static void test_benchmark(void)
{
    ae_calib_t def;
    ae_calib_default(&def);
    volatile float sink = 0.0f; // volatile: el compilador no puede eliminar las inferencias
    for (int i = 0; i < 4; i++) {
        sink += ae_infer(tv_epoch(i % AE_TV_COUNT), &def, &g_ws).score; // calentar caches
    }
    const int64_t t0 = now_us();
    for (int i = 0; i < BENCH_ITERS; i++) {
        sink += ae_infer(tv_epoch(i % AE_TV_COUNT), &def, &g_ws).score;
    }
    const int64_t t1 = now_us();
    const double us = (double)(t1 - t0) / BENCH_ITERS;
    printf("  info: benchmark %d x ae_infer: %.1f us/epoca (%s; sink %.3g)\n", BENCH_ITERS, us, RUNNER_KIND,
           (double)sink);
#if defined(ESP_PLATFORM) && !defined(AE_TEST_FAKE_RUNNER)
    CHECK(us < AE_BUDGET_US, "%.1f us por epoca: supera el presupuesto de 20 ms del CLAUDE.md", us);
#endif
}

// ---------------------------------------------------------------- runner de tests

typedef struct {
    void (*fn)(void);
    const char *name;
    int line;
} test_case_t;

#define TEST_ENTRY(fn) {fn, #fn, __LINE__}

static const test_case_t TESTS[] = {
    TEST_ENTRY(test_not_initialised), // primero: antes de cualquier ae_init
    TEST_ENTRY(test_init),
    TEST_ENTRY(test_preprocess_goldens),
    TEST_ENTRY(test_preprocess_structure),
    TEST_ENTRY(test_normalize),
    TEST_ENTRY(test_quantize),
    TEST_ENTRY(test_dequantize),
    TEST_ENTRY(test_score),
    TEST_ENTRY(test_levels),
    TEST_ENTRY(test_infer_goldens),
    TEST_ENTRY(test_golden_coverage),
    TEST_ENTRY(test_fail_safe),
#ifdef AE_TEST_FAKE_RUNNER
    TEST_ENTRY(test_runner_faults),
#else
    TEST_ENTRY(test_runner_real),
#endif
    TEST_ENTRY(test_calibration_goldens),
    TEST_ENTRY(test_calibration_reference),
    TEST_ENTRY(test_calibration_flow),
    TEST_ENTRY(test_determinism),
    TEST_ENTRY(test_workspace_reuse),
    TEST_ENTRY(test_benchmark),
};

static int run_all_tests(void)
{
    g_file = base_name(__FILE__);
    printf("\nae_test: vectores de %s: %d epocas, %d ventanas, %d epocas + %d scores de calibracion (%s)\n",
           AE_TV_MODEL_ID, AE_TV_COUNT, AE_TV_N_WIN, AE_TV_CAL_N_EPOCHS, AE_TV_CAL_N_SCORES, RUNNER_KIND);

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
    // Tarea propia: la pila de main (3.5 KB por defecto) es justa para printf + Invoke de TFLM
    xTaskCreate(test_task, "ae_test", 16384, NULL, 5, NULL);
}
#else
int main(void)
{
    return run_all_tests() == 0 ? 0 : 1;
}
#endif
