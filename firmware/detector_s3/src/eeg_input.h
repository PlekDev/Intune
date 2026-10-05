// Entrada de EEG desde el puente (C1): UART -> muestras filtradas en el ring buffer.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "errp_dsp.h"

typedef struct {
    uint32_t frames, gaps, lost, filter_resets, corrupt, discarded, bytes;
    float battery_pct;
    bool t0_valid;
} eeg_input_stats_t;

void eeg_input_start(errp_ring_t *ring, SemaphoreHandle_t ring_lock);

// Contador Unicorn de la muestra adquirida en t_us (esp_timer), con t0 = envolvente
// inferior de (t_llegada - contador x 4 ms). false si aún no hay estimación.
bool eeg_input_counter_at(int64_t t_us, uint32_t *counter);

// true si llegan tramas válidas y el puente dice que está en streaming.
bool eeg_input_ok(int64_t now_us);

void eeg_input_get_stats(eeg_input_stats_t *out);
