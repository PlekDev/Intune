// Parser de tramas Unicorn con resincronización y validación de contador.
// Header-only, sin dependencias de ESP-IDF: lo usan el puente (monitor pasivo)
// y el receptor S3. Alimentar con bytes arbitrarios (ráfagas BT/UART).
//
// Resync: ventana de 45 B; válida si C0 00 ... 0D 0A. Si no, avanzar 1 byte.
// Una ventana puede pasar el chequeo por casualidad (trama truncada + siguiente):
// saltos de contador > UNICORN_MAX_GAP se rechazan como corruptos; tras
// UNICORN_RESYNC_AFTER rechazos seguidos se acepta el contador nuevo (reinicio real).
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "unicorn_protocol.h"

#define UNICORN_MAX_GAP      (10 * UNICORN_FS_HZ)
#define UNICORN_RESYNC_AFTER 3

typedef struct {
    uint32_t counter;
    float eeg_uv[UNICORN_N_EEG];
    float acc_g[3];
    float gyr_dps[3];
    float battery_pct;
} unicorn_sample_t;

typedef struct {
    uint8_t win[UNICORN_FRAME_LEN];
    int fill;
    bool have_prev;
    uint32_t prev_cnt;
    // Stats acumuladas
    uint32_t frames;      // tramas válidas
    uint32_t gaps;        // eventos de hueco
    uint32_t lost;        // muestras faltantes (suma de huecos)
    uint32_t backwards;   // contador repetido o que retrocede (p.ej. reinicio de sesión)
    uint32_t discarded;   // bytes descartados buscando sincronía
    uint32_t corrupt;     // tramas con forma válida pero contador imposible (rechazadas)
    int bad_run;          // rechazos consecutivos
    uint32_t last_gap;    // tamaño del último hueco
} unicorn_parser_t;

// Callback por trama válida. gap = muestras faltantes justo antes de esta (0 si consecutiva).
typedef void (*unicorn_frame_cb_t)(const uint8_t *frame, uint32_t gap, void *ctx);

static inline void unicorn_parser_init(unicorn_parser_t *p)
{
    memset(p, 0, sizeof(*p));
}

// Olvida el contador previo (nueva sesión de streaming) sin borrar stats.
static inline void unicorn_parser_new_session(unicorn_parser_t *p)
{
    p->fill = 0;
    p->have_prev = false;
}

static inline void unicorn_decode(const uint8_t *f, unicorn_sample_t *s)
{
    s->counter = unicorn_counter(f);
    s->battery_pct = unicorn_battery_pct(f);
    for (int ch = 0; ch < UNICORN_N_EEG; ch++) {
        s->eeg_uv[ch] = unicorn_eeg_raw(f, ch) * UNICORN_EEG_SCALE_UV;
    }
    for (int i = 0; i < 3; i++) {
        s->acc_g[i] = unicorn_i16le(f + UNICORN_OFF_ACC + 2 * i) * UNICORN_ACC_SCALE_G;
        s->gyr_dps[i] = unicorn_i16le(f + UNICORN_OFF_GYR + 2 * i) * UNICORN_GYR_SCALE_DPS;
    }
}

static inline void unicorn_parser_feed(unicorn_parser_t *p, const uint8_t *data, size_t len,
                                       unicorn_frame_cb_t cb, void *ctx)
{
    for (size_t i = 0; i < len; i++) {
        p->win[p->fill++] = data[i];
        if (p->fill < UNICORN_FRAME_LEN) {
            // Descartar pronto si la cabecera ya no cuadra (evita esperar 45 B en vano)
            if (p->fill == 1 && p->win[0] != UNICORN_HDR0) {
                p->fill = 0;
                p->discarded++;
            } else if (p->fill == 2 && p->win[1] != UNICORN_HDR1) {
                // win[1] podría ser el inicio de la siguiente cabecera
                p->discarded++;
                p->win[0] = p->win[1];
                p->fill = (p->win[0] == UNICORN_HDR0) ? 1 : 0;
                if (p->fill == 0) {
                    p->discarded++;
                }
            }
            continue;
        }
        if (!unicorn_frame_valid_at(p->win)) {
            // Avanzar 1 byte y volver a buscar cabecera dentro de lo ya recibido
            p->discarded++;
            uint8_t tmp[UNICORN_FRAME_LEN - 1];
            int n = UNICORN_FRAME_LEN - 1;
            memcpy(tmp, p->win + 1, n);
            p->fill = 0;
            unicorn_parser_feed(p, tmp, n, cb, ctx); // recursión acotada: tmp < 45 B
            continue;
        }
        uint32_t cnt = unicorn_counter(p->win);
        uint32_t gap = 0;
        if (p->have_prev) {
            uint32_t diff = cnt - p->prev_cnt;
            bool implausible = diff == 0 || diff > UNICORN_MAX_GAP;
            if (implausible && ++p->bad_run < UNICORN_RESYNC_AFTER) {
                p->corrupt++;
                p->fill = 0;
                continue;
            }
            p->bad_run = 0;
            if (implausible) {
                p->backwards++; // contador reiniciado de verdad: reenganchar sin contar hueco
            } else if (diff > 1) {
                gap = diff - 1;
                p->gaps++;
                p->lost += gap;
                p->last_gap = gap;
            }
        }
        p->prev_cnt = cnt;
        p->have_prev = true;
        p->frames++;
        if (cb) {
            cb(p->win, gap, ctx);
        }
        p->fill = 0;
    }
}

// Porcentaje de pérdida acumulado
static inline float unicorn_parser_loss_pct(const unicorn_parser_t *p)
{
    uint32_t expected = p->frames + p->lost;
    return expected ? 100.0f * p->lost / expected : 0.0f;
}
