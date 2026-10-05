// Enlace UART ESP32 clásico (C1, TX) -> ESP32-S3 (C2, RX). 921600 8N1.
// Header-only, sin dependencias de ESP-IDF: lo usan ambos extremos.
// BORRADOR v1 (C1). C2 debe validarlo antes de congelarlo.
//
// Trama (little-endian, sin padding):
//   off  tam  campo
//   0    2    sync  A5 5A
//   2    1    tipo  (LINK_TYPE_*)
//   3    1    len   bytes de payload
//   4    len  payload
//   4+len 2   CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) sobre [tipo .. fin del payload]
//
// EEG: una trama por muestra Unicorn (250/s), 55 B -> 13.75 kB/s (~15 % de 921600).
// STATUS: 1/s siempre, también sin streaming; es el heartbeat del puente.
// EEG_RAW (solo depuración, Kconfig): mismo payload que EEG pero en µV sin filtrar,
// enviado justo después de cada EEG. Sirve para comparar el IIR con scipy (prueba 7).
// EVENT: flanco de subida del pulso de sincronía del brazo, si llega también al puente
// (Kconfig BRIDGE_SYNC_GPIO). Lo usa la grabación de datos (tools/recording/) mientras
// no existe el S3. Ver "Pulso de sincronía" más abajo.
// El S3 debe ignorar tipos que no conoce.
//
// Huecos (regla dura 3): nunca se interpola.
//   - Falta 1 muestra: se rellena con retención de orden cero (último valor), flag HELD.
//   - Faltan 2..LINK_MAX_HOLD: se rellenan igual, flags HELD|GAP en cada una.
//   - Faltan > LINK_MAX_HOLD: no se rellena; se reinicia el IIR y se envía con FILTER_RESET|SETTLING.
//   SETTLING dura 2 s (EEG_IIR_SETTLE_SAMPLES) tras cada reinicio del filtro.
//   C2 y C4 descartan toda época con alguna muestra HELD, GAP o SETTLING (LINK_F_REJECT).
// Así el contador que ve el S3 es continuo salvo en huecos largos.
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define LINK_BAUD        921600
#define LINK_SYNC0       0xA5
#define LINK_SYNC1       0x5A
#define LINK_N_CH        8       // Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 (orden del Unicorn)
#define LINK_HDR_LEN     4       // sync(2) + tipo + len
#define LINK_CRC_LEN     2
#define LINK_MAX_PAYLOAD 64
#define LINK_MAX_FRAME   (LINK_HDR_LEN + LINK_MAX_PAYLOAD + LINK_CRC_LEN)
#define LINK_MAX_HOLD    25      // muestras (100 ms) que se rellenan con ZOH

enum {
    LINK_TYPE_EEG    = 0x01,
    LINK_TYPE_STATUS = 0x02,
    LINK_TYPE_EEG_RAW = 0x03,  // depuración
    LINK_TYPE_EVENT  = 0x04,
};

// Flags por muestra (LINK_TYPE_EEG)
#define LINK_F_HELD          0x01  // rellenada con ZOH (no viene del Unicorn)
#define LINK_F_GAP           0x02  // parte de un hueco >= 2 muestras: descartar la época
#define LINK_F_SETTLING      0x04  // transitorio del IIR tras inicio/reinicio: descartar la época
#define LINK_F_FILTER_RESET  0x08  // primera muestra tras reiniciar el IIR (discontinuidad)
#define LINK_F_SESSION_START 0x10  // primera muestra de una sesión Unicorn (contador vuelve a 1)
#define LINK_F_UNFILTERED    0x20  // µV sin IIR (opción Kconfig de comparación)

// Muestras inservibles para una época ErrP (puerta de artefactos de CLAUDE.md)
#define LINK_F_REJECT (LINK_F_HELD | LINK_F_GAP | LINK_F_SETTLING)

// Estado del puente (LINK_TYPE_STATUS)
enum {
    LINK_STATE_IDLE       = 0,  // parado (botón BOOT)
    LINK_STATE_CONNECTING = 1,  // inquiry / SDP / conexión / esperando ACK
    LINK_STATE_STREAMING  = 2,
};

typedef struct __attribute__((packed)) {
    uint32_t counter;            // contador original del Unicorn (o el que falta, si HELD)
    uint8_t  flags;              // LINK_F_*
    float    eeg_uv[LINK_N_CH];  // µV, filtrado 1–15 Hz salvo LINK_F_UNFILTERED
    int16_t  acc[3];             // crudo del Unicorn, /4096 -> g (UNICORN_ACC_SCALE_G)
    int16_t  gyr[3];             // crudo del Unicorn, /32.8 -> °/s (UNICORN_GYR_SCALE_DPS); puerta de movimiento
} link_eeg_t;                    // 49 B

typedef struct __attribute__((packed)) {
    uint8_t  state;              // LINK_STATE_*
    uint8_t  battery_pct;        // 0..100, 0xFF = desconocido
    uint16_t proc_us_max;        // máx. µs de proceso por muestra en el último segundo
    uint32_t frames;             // muestras válidas recibidas del Unicorn (acumulado)
    uint32_t gaps;               // eventos de hueco
    uint32_t lost;               // muestras perdidas en total
    uint32_t reconnects;
} link_status_t;                 // 20 B

// Pulso de sincronía (regla dura 4). Mismo método que debe usar el S3:
//   t0 = envolvente inferior de (t_llegada - contador * 4000 us), mínimo móvil
//   muestra del flanco = (t_flanco - t0) / 4000 us, redondeado
// t0 corresponde a la latencia mínima BT; el resto (constante) es EVENT_LATENCY_OFFSET
// y se mide en la prueba 10. counter puede ir por delante de la última muestra recibida.
#define LINK_EVT_F_NO_T0   0x01  // sin streaming o sin t0 todavía: counter no es válido
#define LINK_EVT_F_OVERLAP 0x02  // menos de 1.0 s desde el flanco anterior

typedef struct __attribute__((packed)) {
    uint32_t seq;                // nº de flanco desde el arranque del puente (detecta pérdidas)
    uint32_t counter;            // muestra Unicorn estimada en el flanco
    int16_t  offset_us;          // t_flanco - (t0 + counter * 4000), en [-2000, 2000]
    uint8_t  flags;              // LINK_EVT_F_*
} link_event_t;                  // 11 B

_Static_assert(sizeof(float) == 4, "float de 32 bits");
_Static_assert(sizeof(link_event_t) == 11, "link_event_t");
_Static_assert(sizeof(link_eeg_t) == 49, "link_eeg_t");
_Static_assert(sizeof(link_status_t) == 20, "link_status_t");
_Static_assert(sizeof(link_eeg_t) <= LINK_MAX_PAYLOAD, "payload EEG");

#define LINK_EEG_FRAME_LEN    (LINK_HDR_LEN + sizeof(link_eeg_t) + LINK_CRC_LEN)     // 55
#define LINK_STATUS_FRAME_LEN (LINK_HDR_LEN + sizeof(link_status_t) + LINK_CRC_LEN)  // 26

static inline uint16_t link_crc16(const uint8_t *p, size_t n)
{
    uint16_t crc = 0xFFFF;
    while (n--) {
        crc ^= (uint16_t)(*p++) << 8;
        for (int i = 0; i < 8; i++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

// Arma una trama en out (>= LINK_MAX_FRAME). Devuelve su longitud, 0 si len no cabe.
static inline size_t link_encode(uint8_t *out, uint8_t type, const void *payload, uint8_t len)
{
    if (len > LINK_MAX_PAYLOAD)
        return 0;
    out[0] = LINK_SYNC0;
    out[1] = LINK_SYNC1;
    out[2] = type;
    out[3] = len;
    memcpy(out + LINK_HDR_LEN, payload, len);
    uint16_t crc = link_crc16(out + 2, 2 + (size_t)len);
    out[LINK_HDR_LEN + len] = (uint8_t)crc;
    out[LINK_HDR_LEN + len + 1] = (uint8_t)(crc >> 8);
    return LINK_HDR_LEN + (size_t)len + LINK_CRC_LEN;
}

static inline size_t link_encode_eeg(uint8_t *out, const link_eeg_t *s)
{
    return link_encode(out, LINK_TYPE_EEG, s, sizeof(*s));
}

static inline size_t link_encode_eeg_raw(uint8_t *out, const link_eeg_t *s)
{
    return link_encode(out, LINK_TYPE_EEG_RAW, s, sizeof(*s));
}

static inline size_t link_encode_event(uint8_t *out, const link_event_t *e)
{
    return link_encode(out, LINK_TYPE_EVENT, e, sizeof(*e));
}

static inline size_t link_encode_status(uint8_t *out, const link_status_t *s)
{
    return link_encode(out, LINK_TYPE_STATUS, s, sizeof(*s));
}

// ---- Receptor (S3) ----
// Alimentar byte a byte. Resync: busca A5 5A; si len o CRC fallan, vuelve a buscar sync.

typedef struct {
    uint8_t buf[LINK_MAX_FRAME];
    int fill;
    uint32_t frames;     // tramas con CRC correcto
    uint32_t crc_errors;
    uint32_t discarded;  // bytes descartados buscando sync
} link_rx_t;

// Callback por trama válida: payload apunta dentro del buffer del receptor (copiar si se guarda).
typedef void (*link_frame_cb_t)(uint8_t type, const uint8_t *payload, uint8_t len, void *ctx);

static inline void link_rx_init(link_rx_t *r)
{
    memset(r, 0, sizeof(*r));
}

static inline void link_rx_byte(link_rx_t *r, uint8_t b, link_frame_cb_t cb, void *ctx)
{
    if (r->fill == 0) {
        if (b == LINK_SYNC0)
            r->buf[r->fill++] = b;
        else
            r->discarded++;
        return;
    }
    if (r->fill == 1) {
        if (b == LINK_SYNC1) {
            r->buf[r->fill++] = b;
        } else {
            r->discarded++;
            r->fill = (b == LINK_SYNC0) ? 1 : 0;
        }
        return;
    }
    r->buf[r->fill++] = b;
    if (r->fill == LINK_HDR_LEN && r->buf[3] > LINK_MAX_PAYLOAD) {
        r->discarded += LINK_HDR_LEN;
        r->fill = 0;
        return;
    }
    if (r->fill < LINK_HDR_LEN)
        return;
    int total = LINK_HDR_LEN + r->buf[3] + LINK_CRC_LEN;
    if (r->fill < total)
        return;
    uint16_t got = (uint16_t)(r->buf[total - 2] | (r->buf[total - 1] << 8));
    if (got == link_crc16(r->buf + 2, 2 + (size_t)r->buf[3])) {
        r->frames++;
        if (cb)
            cb(r->buf[2], r->buf + LINK_HDR_LEN, r->buf[3], ctx);
    } else {
        // Un byte A5 5A dentro de una trama perdida puede ser el inicio real: no se
        // re-escanea el buffer; el CRC de la siguiente trama resincroniza.
        r->crc_errors++;
    }
    r->fill = 0;
}

static inline void link_rx_feed(link_rx_t *r, const uint8_t *data, size_t n, link_frame_cb_t cb, void *ctx)
{
    for (size_t i = 0; i < n; i++)
        link_rx_byte(r, data[i], cb, ctx);
}

// Copia segura (sin accesos desalineados) del payload a la estructura.
static inline bool link_decode_eeg(const uint8_t *payload, uint8_t len, link_eeg_t *out)
{
    if (len != sizeof(*out))
        return false;
    memcpy(out, payload, sizeof(*out));
    return true;
}

static inline bool link_decode_status(const uint8_t *payload, uint8_t len, link_status_t *out)
{
    if (len != sizeof(*out))
        return false;
    memcpy(out, payload, sizeof(*out));
    return true;
}
