#include "errp_model.h"

#include <cmath>

#include "errp_dsp.h"
#include "errp_model_data.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "errp_model";

// Activaciones: conv ~8x40x8 int8 + densas; 32 KB con margen. Se reporta lo usado.
constexpr size_t kArenaSize = 32 * 1024;
alignas(16) static uint8_t s_arena[kArenaSize];

static tflite::MicroInterpreter *s_interp;
static TfLiteTensor *s_in, *s_out;
static int64_t s_last_us;

bool errp_model_init(void)
{
    const tflite::Model *model = tflite::GetModel(g_errp_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "versión de esquema %lu != %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    // Solo las ops que puede usar cualquiera de las dos arquitecturas
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();

    static tflite::MicroInterpreter interp(model, resolver, s_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors falló (arena %u B)", (unsigned)kArenaSize);
        return false;
    }
    s_interp = &interp;
    s_in = interp.input(0);
    s_out = interp.output(0);
    if (s_in->type != kTfLiteInt8 || s_out->type != kTfLiteInt8 ||
        s_in->bytes != ERRP_N_CH * ERRP_N_T || s_out->bytes != ERRP_N_CH * ERRP_N_T) {
        ESP_LOGE(TAG, "tensores inesperados: in tipo %d %u B, out tipo %d %u B",
                 s_in->type, (unsigned)s_in->bytes, s_out->type, (unsigned)s_out->bytes);
        return false;
    }
    ESP_LOGI(TAG, "modelo %s (%d params, %u B%s), arena usada %u / %u B, in q(%.5f, %ld) out q(%.5f, %ld)",
             ERRP_MODEL_ARCH, ERRP_MODEL_PARAMS, (unsigned)g_errp_model_len,
             ERRP_MODEL_SYNTHETIC ? ", SINTÉTICO" : "", (unsigned)interp.arena_used_bytes(), (unsigned)kArenaSize,
             s_in->params.scale, (long)s_in->params.zero_point, s_out->params.scale, (long)s_out->params.zero_point);
    return true;
}

float errp_model_score(const float e[ERRP_N_CH][ERRP_N_T], float recon[ERRP_N_CH][ERRP_N_T])
{
    if (!s_interp) {
        return NAN;
    }
    int64_t t = esp_timer_get_time();
    const float in_s = s_in->params.scale;
    const int in_z = s_in->params.zero_point;
    int8_t *q = s_in->data.int8;
    for (int c = 0; c < ERRP_N_CH; c++) {  // NHWC [1, 8, 40, 1]: índice c * 40 + t
        for (int k = 0; k < ERRP_N_T; k++) {
            long v = std::lround(e[c][k] / in_s) + in_z;
            q[c * ERRP_N_T + k] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
        }
    }
    if (s_interp->Invoke() != kTfLiteOk) {
        return NAN;
    }
    const float out_s = s_out->params.scale;
    const int out_z = s_out->params.zero_point;
    const int8_t *o = s_out->data.int8;
    float r[ERRP_N_CH][ERRP_N_T];
    for (int c = 0; c < ERRP_N_CH; c++) {
        for (int k = 0; k < ERRP_N_T; k++) {
            r[c][k] = (o[c * ERRP_N_T + k] - out_z) * out_s;
            if (recon) {
                recon[c][k] = r[c][k];
            }
        }
    }
    float score = errp_mse(e, r);  // contra el epoch en float, no el cuantizado
    s_last_us = esp_timer_get_time() - t;
    return score;
}

size_t errp_model_arena_used(void)
{
    return s_interp ? s_interp->arena_used_bytes() : 0;
}

int64_t errp_model_last_us(void)
{
    return s_last_us;
}
