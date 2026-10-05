// Protocolo S3 (C2) <-> brazo (C3) por UART + línea GPIO de sincronía.
// BORRADOR de C2, PENDIENTE de acuerdo con C3 (co-dueños). No cambiar sin avisar.
// Header-only, sin ESP-IDF: lo usan ambos lados.
//
// UART: 115200 8N1, S3 TX -> brazo RX y brazo TX -> S3 RX, GND común.
// Trama: SOF(0xA5) | type | len | payload[len] | crc16 LE
//        crc16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) sobre type, len y payload.
//
// S3 -> brazo:
//   ARM_MSG_ALERT cada ARM_HEARTBEAT_MS y además en cuanto cambia el nivel.
//   Cada ALERT es también el heartbeat: si el brazo no recibe una trama válida
//   en ARM_WATCHDOG_MS, pasa a nivel 3 (paro seguro) por su cuenta.
// Brazo -> S3:
//   ARM_MSG_EVENT tras cada pulso de sincronía: metadatos de la acción. NUNCA
//   fija el tiempo (lo fija el flanco GPIO).
//   ARM_MSG_CONFIRM cuando el operador pulsa CONFIRM en el brazo.
//
// GPIO de sincronía (regla dura 4): flanco de subida al inicio exacto de cada
// acción (t = 0), ancho >= ARM_SYNC_MIN_PULSE_US, separación >= 1.0 s (si no,
// el S3 marca OVERLAP). Pines: brazo ARM_SYNC_GPIO_ARM -> S3 CONFIG_DETECTOR_SYNC_GPIO.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ARM_UART_BAUD          115200
#define ARM_HEARTBEAT_MS       100
#define ARM_WATCHDOG_MS        300
#define ARM_CONFIRM_TIMEOUT_MS 10000  // propuesta: nivel 2 sin CONFIRM en 10 s -> 3
#define ARM_SYNC_MIN_PULSE_US  100
#define ARM_SYNC_MIN_SPACING_MS 1000

#define ARM_SOF 0xA5
#define ARM_MAX_PAYLOAD 16

enum {
    ARM_MSG_ALERT = 0x01,    // S3 -> brazo
    ARM_MSG_EVENT = 0x10,    // brazo -> S3
    ARM_MSG_CONFIRM = 0x11,  // brazo -> S3
};

// Niveles (CLAUDE.md, Alert levels)
enum {
    ARM_LEVEL_NORMAL = 0,    // velocidad nominal
    ARM_LEVEL_MILD = 1,      // baja velocidad
    ARM_LEVEL_MODERATE = 2,  // pausa, requiere CONFIRM
    ARM_LEVEL_SEVERE = 3,    // paro seguro
};

// Flags de ALERT
#define ARM_FLAG_EPOCH_REJECTED 0x01  // el último epoch no se puntuó (artefacto/hueco)
#define ARM_FLAG_CALIBRATING    0x02  // el S3 está calibrando umbrales
#define ARM_FLAG_EEG_LOST       0x04  // sin EEG válido: nivel 3 por fail-safe
#define ARM_FLAG_OVERLAP        0x08  // pulsos de sincronía a < 1.0 s

typedef struct __attribute__((packed)) {
    uint16_t seq;        // incrementa en cada ALERT
    uint8_t level;       // 0..3
    uint8_t flags;       // ARM_FLAG_*
    uint16_t action_id;  // acción del último epoch evaluado (0xFFFF = ninguna)
    uint16_t score_q;    // score MSE x 1000, saturado a 65535 (solo log/dashboard)
} arm_alert_t;

typedef struct __attribute__((packed)) {
    uint16_t action_id;
    uint8_t action_type;
    uint8_t deliberate_error;  /* LOG ONLY: el S3 jamás lo usa en la lógica de alertas */
} arm_event_t;

typedef struct __attribute__((packed)) {
    uint16_t action_id;  // acción en pausa que se confirma
} arm_confirm_t;

static inline uint16_t arm_crc16(const uint8_t *p, size_t n)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < n; i++) {
        crc ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

// Escribe la trama en out (tamaño >= len + 5). Devuelve los bytes escritos.
static inline size_t arm_encode(uint8_t type, const void *payload, uint8_t len, uint8_t *out)
{
    out[0] = ARM_SOF;
    out[1] = type;
    out[2] = len;
    for (uint8_t i = 0; i < len; i++) {
        out[3 + i] = ((const uint8_t *)payload)[i];
    }
    uint16_t crc = arm_crc16(out + 1, (size_t)len + 2);
    out[3 + len] = (uint8_t)(crc & 0xFF);
    out[4 + len] = (uint8_t)(crc >> 8);
    return (size_t)len + 5;
}

// Decodificador byte a byte con resincronización por SOF.
typedef struct {
    uint8_t buf[ARM_MAX_PAYLOAD + 5];
    uint8_t fill;
    uint32_t frames, crc_errors;
} arm_decoder_t;

// Devuelve true cuando hay una trama completa y válida en d->buf (type = buf[1],
// len = buf[2], payload = buf + 3).
static inline bool arm_decode_byte(arm_decoder_t *d, uint8_t b)
{
    if (d->fill == 0 && b != ARM_SOF) {
        return false;
    }
    d->buf[d->fill++] = b;
    if (d->fill == 3 && d->buf[2] > ARM_MAX_PAYLOAD) {
        d->fill = 0;  // longitud imposible: buscar el siguiente SOF
        return false;
    }
    if (d->fill < 3 || d->fill < d->buf[2] + 5) {
        return false;
    }
    uint8_t len = d->buf[2];
    uint16_t rx = (uint16_t)(d->buf[3 + len] | (d->buf[4 + len] << 8));
    d->fill = 0;
    if (rx != arm_crc16(d->buf + 1, (size_t)len + 2)) {
        d->crc_errors++;
        return false;
    }
    d->frames++;
    return true;
}
