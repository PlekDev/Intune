// Enlace UART ESP32-S3 (C2) <-> supervisor del brazo (C3). 115200 8N1, más un pulso GPIO de sync.
// Header-only, sin dependencias de ESP-IDF: lo usan ambos extremos.
// BORRADOR v1 (C3). C2 debe validarlo antes de congelarlo. Mismo esquema de trama que
// link_protocol.h (C1), con otro sync para que un cable cruzado no se interprete.
//
// Trama (little-endian, sin padding):
//   off  tam  campo
//   0    2    sync  C3 3C
//   2    1    tipo  (ARM_TYPE_*)
//   3    1    len   bytes de payload
//   4    len  payload
//   4+len 2   CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) sobre [tipo .. fin del payload]
//
// S3 -> supervisor
//   ALERT  cada ARM_HEARTBEAT_MS y además en cuanto cambia el nivel. Es el heartbeat:
//          si el supervisor no recibe un ALERT válido en ARM_HEARTBEAT_TIMEOUT_MS, para el brazo
//          como en el nivel 3 (fail-safe). El S3 manda nivel >= 2 con ARM_REASON_EEG_LOST si
//          deja de recibir EEG válido.
// supervisor -> S3
//   EVENT  una por acción del brazo, justo después del flanco de subida del pulso de sync.
//          Solo metadatos: el instante t = 0 lo da el pulso GPIO (regla dura 4), nunca esta trama.
//   STATUS 1/s y en cuanto cambia el estado de seguridad. Heartbeat del supervisor (dashboard).
// Cada extremo ignora los tipos que no conoce.
//
// Pulso de sync (regla dura 4): GPIO del supervisor (ARM_SUP_SYNC_GPIO) -> pin de interrupción del S3.
//   - Activo en ALTO: la línea está en BAJO en reposo y sube a 3.3 V durante ARM_SYNC_PULSE_US.
//     El FLANCO DE SUBIDA es t = 0: el instante en que el supervisor envía al brazo el primer paso
//     de la acción. El S3 configura interrupción por flanco de subida (con pull-down: antes de que
//     arranque el supervisor la línea flota) y en la ISR solo toma esp_timer_get_time().
//   - Separación mínima entre pulsos ARM_SYNC_MIN_SPACING_MS (el supervisor no emite pulsos más
//     seguidos; el S3 marca OVERLAP por debajo de 1.0 s). En grabación (C4) >= 1.5 s.
//   - Los movimientos que no son acciones (home, retomar tras pausa) no generan pulso.
//   - Masa común entre placas. Pin del S3 lo fija C2.
//
// Niveles (el S3 decide el nivel; el supervisor aplica la reacción):
//   0 normal    velocidad nominal
//   1 leve      velocidad reducida
//   2 moderado  pausa; sigue en pausa aunque el nivel baje, hasta que el operador confirma
//               en el supervisor (botón) y el nivel recibido es <= 1
//   3 grave     parada segura; queda enclavada hasta rearme del operador (pulsación larga)
//               con nivel recibido <= 1 y heartbeat vivo
// Parada = mantener la posición con torque (nunca quitar el torque: el brazo cae).
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define ARM_BAUD                 115200
#define ARM_SYNC0                0xC3
#define ARM_SYNC1                0x3C
#define ARM_HDR_LEN              4       // sync(2) + tipo + len
#define ARM_CRC_LEN              2
#define ARM_MAX_PAYLOAD          32
#define ARM_MAX_FRAME            (ARM_HDR_LEN + ARM_MAX_PAYLOAD + ARM_CRC_LEN)

#define ARM_HEARTBEAT_MS         100     // periodo de ALERT
#define ARM_HEARTBEAT_TIMEOUT_MS 300     // 3 ALERT perdidos -> parada
#define ARM_STATUS_PERIOD_MS     1000
#define ARM_SYNC_PULSE_US        1000    // ancho del pulso (alto)
#define ARM_SYNC_MIN_SPACING_MS  1000

// Pines propuestos (supervisor = ESP32 DevKit clásico). Los del S3 los fija C2.
#define ARM_SUP_UART_TX_GPIO     27      // -> RX del S3 (16/17 los usa el enlace con el brazo)
#define ARM_SUP_UART_RX_GPIO     26      // <- TX del S3
#define ARM_SUP_SYNC_GPIO        25      // -> pin de interrupción del S3

enum {
    ARM_TYPE_ALERT  = 0x10,  // S3 -> supervisor
    ARM_TYPE_EVENT  = 0x20,  // supervisor -> S3
    ARM_TYPE_STATUS = 0x21,  // supervisor -> S3
};

enum {
    ARM_LEVEL_NORMAL   = 0,
    ARM_LEVEL_MILD     = 1,
    ARM_LEVEL_MODERATE = 2,
    ARM_LEVEL_SEVERE   = 3,
};

// Por qué el S3 manda ese nivel (para logs y dashboard; la reacción depende solo del nivel).
enum {
    ARM_REASON_NONE     = 0,
    ARM_REASON_ERRP     = 1,  // ErrP detectado (error de reconstrucción del autoencoder)
    ARM_REASON_EEG_LOST = 2,  // sin EEG válido del puente (fail-safe del S3)
    ARM_REASON_BOOT     = 3,  // el S3 acaba de arrancar / modelo sin cargar
    ARM_REASON_MANUAL   = 4,  // forzado a mano (pruebas, demo)
};

typedef struct __attribute__((packed)) {
    uint8_t  level;     // ARM_LEVEL_*
    uint8_t  reason;    // ARM_REASON_*
    uint16_t seq;       // +1 por trama (con vuelta): el supervisor cuenta pérdidas
    uint32_t t_ms;      // reloj del S3, informativo
    float    score;     // error de reconstrucción del autoencoder; NaN si no aplica
} arm_alert_t;          // 12 B

// Etiqueta de la acción (EVENT). Durante la recogida de datos (C4) algunas acciones son
// errores deliberados; en operación normal todas son CORRECT.
// LOG ONLY: el S3 solo la reenvía al dashboard y a los logs de grabación; nunca la usa para decidir el nivel.
enum {
    ARM_LABEL_CORRECT   = 0,
    ARM_LABEL_ERROR     = 1,  // error deliberado (paradigma ErrP)
    ARM_LABEL_UNLABELED = 2,
};

enum {
    ARM_EVENT_ONSET   = 0,  // inicio de la acción (acompaña al pulso de sync)
    ARM_EVENT_ABORTED = 1,  // la acción se interrumpió (pausa/parada); sin pulso
};

typedef struct __attribute__((packed)) {
    uint32_t seq;          // +1 por EVENT
    uint16_t action_id;    // índice de la acción en la secuencia del supervisor
    uint8_t  kind;         // ARM_EVENT_*
    uint8_t  label;        // ARM_LABEL_*  /* LOG ONLY */: el S3 nunca lo usa en la lógica de alertas
    uint32_t t_us;         // reloj del supervisor en el flanco del pulso (informativo)
    uint8_t  level;        // nivel aplicado en ese momento
} arm_event_t;             // 13 B

// Estado de seguridad del supervisor (STATUS).
enum {
    ARM_SAFETY_NO_S3   = 0,  // sin heartbeat del S3: brazo parado
    ARM_SAFETY_RUN     = 1,  // nivel 0
    ARM_SAFETY_SLOW    = 2,  // nivel 1
    ARM_SAFETY_PAUSED  = 3,  // nivel 2 o pendiente de confirmación
    ARM_SAFETY_STOPPED = 4,  // nivel 3 enclavado, pendiente de rearme
};

typedef struct __attribute__((packed)) {
    uint8_t  safety;         // ARM_SAFETY_*
    uint8_t  level_rx;       // último nivel recibido del S3 (0xFF = ninguno)
    uint8_t  motion;         // estado interno del movimiento (motion_state_t del supervisor)
    uint8_t  flags;          // ARM_STF_*
    uint32_t alerts_rx;      // ALERT válidos recibidos
    uint32_t alerts_lost;    // huecos en seq
    uint32_t crc_errors;
    uint32_t espnow_fail;    // pasos al brazo sin ACK
} arm_status_t;              // 20 B

#define ARM_STF_ARM_LINK_LOST 0x01  // el brazo no confirma por ESP-NOW
#define ARM_STF_POSE_UNKNOWN  0x02  // falta home
#define ARM_STF_ACTIONS_ON    0x04  // reproductor de acciones en marcha

_Static_assert(sizeof(float) == 4, "float de 32 bits");
_Static_assert(sizeof(arm_alert_t) == 12, "arm_alert_t");
_Static_assert(sizeof(arm_event_t) == 13, "arm_event_t");
_Static_assert(sizeof(arm_status_t) == 20, "arm_status_t");
_Static_assert(sizeof(arm_status_t) <= ARM_MAX_PAYLOAD, "payload STATUS");

static inline uint16_t arm_crc16(const uint8_t *p, size_t n)
{
    uint16_t crc = 0xFFFF;
    while (n--) {
        crc ^= (uint16_t)(*p++) << 8;
        for (int i = 0; i < 8; i++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

// Arma una trama en out (>= ARM_MAX_FRAME). Devuelve su longitud, 0 si len no cabe.
static inline size_t arm_encode(uint8_t *out, uint8_t type, const void *payload, uint8_t len)
{
    if (len > ARM_MAX_PAYLOAD)
        return 0;
    out[0] = ARM_SYNC0;
    out[1] = ARM_SYNC1;
    out[2] = type;
    out[3] = len;
    memcpy(out + ARM_HDR_LEN, payload, len);
    uint16_t crc = arm_crc16(out + 2, 2 + (size_t)len);
    out[ARM_HDR_LEN + len] = (uint8_t)crc;
    out[ARM_HDR_LEN + len + 1] = (uint8_t)(crc >> 8);
    return ARM_HDR_LEN + (size_t)len + ARM_CRC_LEN;
}

static inline size_t arm_encode_alert(uint8_t *out, const arm_alert_t *a)
{
    return arm_encode(out, ARM_TYPE_ALERT, a, sizeof(*a));
}

static inline size_t arm_encode_event(uint8_t *out, const arm_event_t *e)
{
    return arm_encode(out, ARM_TYPE_EVENT, e, sizeof(*e));
}

static inline size_t arm_encode_status(uint8_t *out, const arm_status_t *s)
{
    return arm_encode(out, ARM_TYPE_STATUS, s, sizeof(*s));
}

// ---- Receptor (ambos extremos) ----
// Alimentar byte a byte. Resync: busca C3 3C; si len o CRC fallan, vuelve a buscar sync.

typedef struct {
    uint8_t buf[ARM_MAX_FRAME];
    int fill;
    uint32_t frames;     // tramas con CRC correcto
    uint32_t crc_errors;
    uint32_t discarded;  // bytes descartados buscando sync
} arm_rx_t;

// Callback por trama válida: payload apunta dentro del buffer del receptor (copiar si se guarda).
typedef void (*arm_frame_cb_t)(uint8_t type, const uint8_t *payload, uint8_t len, void *ctx);

static inline void arm_rx_init(arm_rx_t *r)
{
    memset(r, 0, sizeof(*r));
}

static inline void arm_rx_byte(arm_rx_t *r, uint8_t b, arm_frame_cb_t cb, void *ctx)
{
    if (r->fill == 0) {
        if (b == ARM_SYNC0)
            r->buf[r->fill++] = b;
        else
            r->discarded++;
        return;
    }
    if (r->fill == 1) {
        if (b == ARM_SYNC1) {
            r->buf[r->fill++] = b;
        } else {
            r->discarded++;
            r->fill = (b == ARM_SYNC0) ? 1 : 0;
        }
        return;
    }
    r->buf[r->fill++] = b;
    if (r->fill == ARM_HDR_LEN && r->buf[3] > ARM_MAX_PAYLOAD) {
        r->discarded += ARM_HDR_LEN;
        r->fill = 0;
        return;
    }
    if (r->fill < ARM_HDR_LEN)
        return;
    int total = ARM_HDR_LEN + r->buf[3] + ARM_CRC_LEN;
    if (r->fill < total)
        return;
    uint16_t got = (uint16_t)(r->buf[total - 2] | (r->buf[total - 1] << 8));
    if (got == arm_crc16(r->buf + 2, 2 + (size_t)r->buf[3])) {
        r->frames++;
        if (cb)
            cb(r->buf[2], r->buf + ARM_HDR_LEN, r->buf[3], ctx);
    } else {
        r->crc_errors++;
    }
    r->fill = 0;
}

static inline void arm_rx_feed(arm_rx_t *r, const uint8_t *data, size_t n, arm_frame_cb_t cb, void *ctx)
{
    for (size_t i = 0; i < n; i++)
        arm_rx_byte(r, data[i], cb, ctx);
}

// Copia segura (sin accesos desalineados) del payload a la estructura.
static inline bool arm_decode_alert(const uint8_t *payload, uint8_t len, arm_alert_t *out)
{
    if (len != sizeof(*out) || payload[0] > ARM_LEVEL_SEVERE)
        return false;
    memcpy(out, payload, sizeof(*out));
    return true;
}

static inline bool arm_decode_event(const uint8_t *payload, uint8_t len, arm_event_t *out)
{
    if (len != sizeof(*out))
        return false;
    memcpy(out, payload, sizeof(*out));
    return true;
}

static inline bool arm_decode_status(const uint8_t *payload, uint8_t len, arm_status_t *out)
{
    if (len != sizeof(*out))
        return false;
    memcpy(out, payload, sizeof(*out));
    return true;
}
