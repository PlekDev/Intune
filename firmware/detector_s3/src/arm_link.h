// Enlace UART con el brazo (C3) según firmware/common/arm_protocol.h.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "arm_protocol.h"

typedef struct {
    uint32_t tx_frames, rx_frames, rx_crc_errors, events, confirms;
} arm_link_stats_t;

void arm_link_start(void);

// Fija el estado que se manda en cada heartbeat; si cambia nivel o flags se envía ya.
void arm_link_set_alert(uint8_t level, uint8_t flags, uint16_t action_id, float score);

// Saca el siguiente EVENT recibido (no bloquea).
bool arm_link_pop_event(arm_event_t *ev);

// true (una sola vez) si llegó un CONFIRM desde la última llamada.
bool arm_link_take_confirm(void);

void arm_link_get_stats(arm_link_stats_t *out);
