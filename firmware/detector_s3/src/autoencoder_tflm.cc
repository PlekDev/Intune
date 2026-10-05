// Runner TensorFlow Lite Micro del ErrP-AE (detector ESP32-S3): implementa la frontera
// ae_runner_init / ae_runner_invoke de autoencoder_engine.h con el modelo int8 de
// autoencoder_weights.h. CONTRACT v3, sección 7.
//
// - Es el ÚNICO archivo que incluye autoencoder_weights.h (AE_MODEL_TFLITE y metadatos
//   static const): una sola copia del modelo en flash.
// - Sin heap y sin constructores globales: arena estática alignas(16) de
//   AE_TENSOR_ARENA_BYTES; resolver e intérprete se construyen con placement new en
//   almacenamiento estático dentro de ae_runner_init (ae_init, al arrancar).
// - MicroMutableOpResolver con exactamente las 6 ops permitidas (CONV_2D, DEPTHWISE_CONV_2D,
//   AVERAGE_POOL_2D, FULLY_CONNECTED, RESHAPE, RELU): un modelo con otra op falla en ae_init
//   (fail-safe), nunca a mitad de una inferencia.
// - Comprueba al iniciar: flatbuffer válido (Verifier) con identificador TFL3, versión de
//   esquema, un solo subgrafo, 1 entrada y 1 salida int8 de forma (1, 8, 40, 1) con
//   cuantización por tensor y scale / zero_point IGUALES (bit a bit) a AE_IN_* / AE_OUT_* del
//   header (el exportador los escribe con ida y vuelta exacta).
// - NO reentrante (un único intérprete): ver autoencoder_engine.h.
// - C++17 compatible con -fno-exceptions -fno-rtti; solo exporta funciones extern "C".
// Los mensajes de error salen por MicroPrintf (DebugLog de TFLM: consola serie en el S3).
#include "autoencoder_engine.h"
#include "autoencoder_weights.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if !defined(AE_MODEL_ID) || !defined(AE_MODEL_ARCH) || !defined(AE_WEIGHTS_PLACEHOLDER) || !defined(INPUT_DIM) || \
    !defined(OUTPUT_DIM) || !defined(AE_MODEL_N_CH) || !defined(AE_MODEL_N_T) || !defined(AE_IN_SCALE) ||           \
    !defined(AE_IN_ZERO_POINT) || !defined(AE_OUT_SCALE) || !defined(AE_OUT_ZERO_POINT) ||                          \
    !defined(AE_TENSOR_ARENA_BYTES) || !defined(AE_INT8_SCORE_CORR) || !defined(THRESHOLD_LEVEL_1) ||               \
    !defined(THRESHOLD_LEVEL_2) || !defined(THRESHOLD_LEVEL_3) || !defined(AE_CALIB_PCT_1) ||                       \
    !defined(AE_CALIB_PCT_2) || !defined(AE_CALIB_PCT_3)
#error "autoencoder_weights.h incompleto o de otra version (v1?): regenerar con ml/c_exporter/export_to_c.py"
#endif

static_assert(INPUT_DIM == AE_N_IN && OUTPUT_DIM == AE_N_IN, "el modelo debe ser 320 -> 320 (8 x 40)");
static_assert(AE_MODEL_N_CH == AE_N_CH && AE_MODEL_N_T == AE_N_T, "el modelo no es de epocas 8 x 40");
static_assert(sizeof(AE_MODEL_ID) == AE_MODEL_ID_LEN + 1, "AE_MODEL_ID debe tener 16 caracteres");
static_assert(sizeof(MEAN_VECTOR) / sizeof(MEAN_VECTOR[0]) == AE_N_CH, "MEAN_VECTOR debe tener 8 canales");
static_assert(sizeof(STD_VECTOR) / sizeof(STD_VECTOR[0]) == AE_N_CH, "STD_VECTOR debe tener 8 canales");
static_assert(AE_TENSOR_ARENA_BYTES > 0, "AE_TENSOR_ARENA_BYTES debe ser > 0");
static_assert((AE_IN_ZERO_POINT) >= -128 && (AE_IN_ZERO_POINT) <= 127, "AE_IN_ZERO_POINT fuera de int8");
static_assert((AE_OUT_ZERO_POINT) >= -128 && (AE_OUT_ZERO_POINT) <= 127, "AE_OUT_ZERO_POINT fuera de int8");

namespace {

constexpr int kAeNumOps = 6;
using AeOpResolver = tflite::MicroMutableOpResolver<kAeNumOps>;

// Arena de TFLM (.bss). Almacenamiento crudo para resolver e intérprete: se construyen con
// placement new en ae_runner_init (sin heap, sin constructores estáticos al arrancar).
alignas(16) uint8_t g_arena[AE_TENSOR_ARENA_BYTES];
alignas(AeOpResolver) unsigned char g_resolver_mem[sizeof(AeOpResolver)];
alignas(tflite::MicroInterpreter) unsigned char g_interp_mem[sizeof(tflite::MicroInterpreter)];

AeOpResolver *g_resolver = nullptr;
tflite::MicroInterpreter *g_interp = nullptr;
int8_t *g_in = nullptr;        // datos del tensor de entrada (dentro de la arena)
const int8_t *g_out = nullptr; // datos del tensor de salida (dentro de la arena)
bool g_ok = false;

void ae_tflm_reset()
{
    if (g_interp != nullptr) {
        g_interp->~MicroInterpreter();
        g_interp = nullptr;
    }
    if (g_resolver != nullptr) {
        g_resolver->~AeOpResolver();
        g_resolver = nullptr;
    }
    g_in = nullptr;
    g_out = nullptr;
    g_ok = false;
}

// Tensor int8 [1, 8, 40, 1] con cuantización por tensor igual a la del header
bool ae_tensor_ok(const TfLiteTensor *t, float scale, int32_t zero_point, const char *what)
{
    if (t == nullptr || t->data.int8 == nullptr) {
        MicroPrintf("ae_tflm: tensor de %s ausente", what);
        return false;
    }
    if (t->type != kTfLiteInt8) {
        MicroPrintf("ae_tflm: tensor de %s de tipo %d, se esperaba int8", what, static_cast<int>(t->type));
        return false;
    }
    const TfLiteIntArray *d = t->dims;
    if (d == nullptr || d->size != 4 || d->data[0] != 1 || d->data[1] != AE_N_CH || d->data[2] != AE_N_T ||
        d->data[3] != 1 || t->bytes != static_cast<size_t>(AE_N_IN)) {
        MicroPrintf("ae_tflm: tensor de %s con forma distinta de (1, %d, %d, 1)", what, AE_N_CH, AE_N_T);
        return false;
    }
    const auto *aq = static_cast<const TfLiteAffineQuantization *>(t->quantization.params);
    if (t->quantization.type != kTfLiteAffineQuantization || aq == nullptr || aq->scale == nullptr ||
        aq->scale->size != 1 || aq->zero_point == nullptr || aq->zero_point->size != 1) {
        MicroPrintf("ae_tflm: tensor de %s sin cuantizacion int8 por tensor", what);
        return false;
    }
    // Igualdad exacta a propósito: el header escribe scale con ida y vuelta float32 exacta
    if (!(t->params.scale == scale)) {
        MicroPrintf("ae_tflm: scale de %s distinta de la del header (modelo y macros de exportaciones "
                    "distintas: regenerar autoencoder_weights.h)",
                    what);
        return false;
    }
    if (t->params.zero_point != zero_point) {
        MicroPrintf("ae_tflm: zero_point de %s %d, header %d (modelo y macros de exportaciones distintas: "
                    "regenerar autoencoder_weights.h)",
                    what, static_cast<int>(t->params.zero_point), static_cast<int>(zero_point));
        return false;
    }
    return true;
}

bool ae_tflm_setup()
{
    flatbuffers::Verifier verifier(AE_MODEL_TFLITE, static_cast<size_t>(AE_MODEL_TFLITE_LEN));
    if (!tflite::VerifyModelBuffer(verifier)) {
        MicroPrintf("ae_tflm: AE_MODEL_TFLITE no es un modelo TFLite valido (flatbuffer TFL3)");
        return false;
    }
    const tflite::Model *model = tflite::GetModel(AE_MODEL_TFLITE);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("ae_tflm: esquema TFLite %d, se esperaba %d", static_cast<int>(model->version()),
                    TFLITE_SCHEMA_VERSION);
        return false;
    }
    if (model->subgraphs() == nullptr || model->subgraphs()->size() != 1) {
        MicroPrintf("ae_tflm: el modelo debe tener un solo subgrafo");
        return false;
    }

    // Exactamente las 6 ops permitidas por el CONTRACT (kernels de referencia o ESP-NN)
    g_resolver = new (g_resolver_mem) AeOpResolver();
    if (g_resolver->AddConv2D() != kTfLiteOk || g_resolver->AddDepthwiseConv2D() != kTfLiteOk ||
        g_resolver->AddAveragePool2D() != kTfLiteOk || g_resolver->AddFullyConnected() != kTfLiteOk ||
        g_resolver->AddReshape() != kTfLiteOk || g_resolver->AddRelu() != kTfLiteOk) {
        MicroPrintf("ae_tflm: no se pudieron registrar las ops");
        return false;
    }

    g_interp = new (g_interp_mem) tflite::MicroInterpreter(model, *g_resolver, g_arena, sizeof(g_arena));
    if (g_interp->initialization_status() != kTfLiteOk) {
        MicroPrintf("ae_tflm: fallo al crear el interprete");
        return false;
    }
    if (g_interp->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("ae_tflm: AllocateTensors fallo (arena de %d B insuficiente u op no permitida)",
                    static_cast<int>(sizeof(g_arena)));
        return false;
    }
    if (g_interp->inputs_size() != 1 || g_interp->outputs_size() != 1) {
        MicroPrintf("ae_tflm: el modelo debe tener 1 entrada y 1 salida");
        return false;
    }
    TfLiteTensor *in = g_interp->input(0);
    const TfLiteTensor *out = g_interp->output(0);
    if (!ae_tensor_ok(in, AE_IN_SCALE, AE_IN_ZERO_POINT, "entrada") ||
        !ae_tensor_ok(out, AE_OUT_SCALE, AE_OUT_ZERO_POINT, "salida")) {
        return false;
    }
    g_in = in->data.int8;
    g_out = out->data.int8;
    g_ok = true;
    return true;
}

} // namespace

extern "C" {

bool ae_runner_init(ae_model_info_t *info)
{
    if (info == nullptr) {
        return false;
    }
    ae_tflm_reset(); // otra llamada reconstruye el intérprete desde cero
    if (!ae_tflm_setup()) {
        ae_tflm_reset();
        return false;
    }
    info->model_id = AE_MODEL_ID;
    info->arch = AE_MODEL_ARCH;
    info->placeholder = (AE_WEIGHTS_PLACEHOLDER) != 0;
    info->in_scale = AE_IN_SCALE;
    info->in_zero_point = AE_IN_ZERO_POINT;
    info->out_scale = AE_OUT_SCALE;
    info->out_zero_point = AE_OUT_ZERO_POINT;
    std::memcpy(info->default_mean, MEAN_VECTOR, sizeof(info->default_mean));
    std::memcpy(info->default_std, STD_VECTOR, sizeof(info->default_std));
    info->default_t[0] = THRESHOLD_LEVEL_1;
    info->default_t[1] = THRESHOLD_LEVEL_2;
    info->default_t[2] = THRESHOLD_LEVEL_3;
    info->calib_pct[0] = AE_CALIB_PCT_1;
    info->calib_pct[1] = AE_CALIB_PCT_2;
    info->calib_pct[2] = AE_CALIB_PCT_3;
    info->arena_bytes = static_cast<uint32_t>(AE_TENSOR_ARENA_BYTES);
    info->arena_used_bytes = static_cast<uint32_t>(g_interp->arena_used_bytes());
    info->int8_score_corr = AE_INT8_SCORE_CORR;
    return true;
}

bool ae_runner_invoke(const int8_t in[AE_N_IN], int8_t out[AE_N_IN])
{
    if (!g_ok || in == nullptr || out == nullptr) {
        return false;
    }
    // La entrada se copia en cada llamada: el planificador de memoria puede reutilizar su
    // buffer durante Invoke
    std::memcpy(g_in, in, AE_N_IN);
    if (g_interp->Invoke() != kTfLiteOk) {
        MicroPrintf("ae_tflm: Invoke fallo");
        return false;
    }
    std::memcpy(out, g_out, AE_N_IN);
    return true;
}

} // extern "C"
