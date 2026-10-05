// Mensaje ESP-NOW que entiende el firmware de fábrica del RoArm-M2-S.
// Copia de struct_message en roarm_ref/src/esp_now_ctrl.h: mismo orden, mismos tipos,
// mismo compilador (xtensa GCC) => mismo layout. El brazo hace memcpy de sizeof(struct),
// así que se envía siempre la estructura completa.
#pragma once
#include <assert.h>
#include <stdint.h>

typedef struct {
    uint8_t devCode;     // no lo usa el brazo
    float base;          // cmd 0: ángulos absolutos (rad) a velocidad MÁXIMA -> no usar
    float shoulder;
    float elbow;
    float hand;
    uint8_t cmd;         // 0 = ángulos de arriba, 1 = JSON en el callback Wi-Fi, 2 = JSON en loop(), 3 = solo imprime
    char message[210];   // JSON, terminado en '\0'
} roarm_espnow_msg_t;

static_assert(sizeof(roarm_espnow_msg_t) == 232, "layout distinto al del firmware del brazo");

#define ROARM_ESPNOW_JSON_IN_CALLBACK 1
#define ROARM_ESPNOW_JSON_IN_LOOP     2
