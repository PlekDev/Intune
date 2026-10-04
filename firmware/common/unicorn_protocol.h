// Constantes del protocolo Bluetooth del g.tec Unicorn Hybrid Black.
// Compartido por el puente (solo comandos) y el receptor S3 (parseo).
// Verificado en Linux 2026-10-03 (tools/linux_probe/unicorn_probe.py).
#pragma once
#include <stdint.h>

// Comandos (host -> Unicorn); ambos responden ACK 00 00 00
static const uint8_t UNICORN_CMD_START[3] = {0x61, 0x7C, 0x87};
static const uint8_t UNICORN_CMD_STOP[3]  = {0x63, 0x5C, 0xC5};
static const uint8_t UNICORN_ACK[3]       = {0x00, 0x00, 0x00};

#define UNICORN_FS_HZ        250
#define UNICORN_FRAME_LEN    45
#define UNICORN_N_EEG        8

// Offsets dentro de la trama
#define UNICORN_OFF_HDR0     0   // 0xC0
#define UNICORN_OFF_HDR1     1   // 0x00
#define UNICORN_OFF_BATT     2   // 100*(b&0x0F)/15 %
#define UNICORN_OFF_EEG      3   // 8 x int24 big-endian
#define UNICORN_OFF_ACC      27  // 3 x int16 little-endian
#define UNICORN_OFF_GYR      33  // 3 x int16 little-endian
#define UNICORN_OFF_CNT      39  // uint32 little-endian
#define UNICORN_OFF_FTR0     43  // 0x0D
#define UNICORN_OFF_FTR1     44  // 0x0A

#define UNICORN_HDR0 0xC0
#define UNICORN_HDR1 0x00
#define UNICORN_FTR0 0x0D
#define UNICORN_FTR1 0x0A

#define UNICORN_EEG_SCALE_UV (4500000.0f / 50331642.0f)
#define UNICORN_ACC_SCALE_G  (1.0f / 4096.0f)
#define UNICORN_GYR_SCALE_DPS (1.0f / 32.8f)

static inline int unicorn_frame_valid_at(const uint8_t *p)
{
    return p[UNICORN_OFF_HDR0] == UNICORN_HDR0 && p[UNICORN_OFF_HDR1] == UNICORN_HDR1 &&
           p[UNICORN_OFF_FTR0] == UNICORN_FTR0 && p[UNICORN_OFF_FTR1] == UNICORN_FTR1;
}

static inline int32_t unicorn_eeg_raw(const uint8_t *frame, int ch)
{
    const uint8_t *b = frame + UNICORN_OFF_EEG + 3 * ch;
    return ((int32_t)(((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8))) >> 8;
}

static inline int16_t unicorn_i16le(const uint8_t *p) { return (int16_t)(p[0] | (p[1] << 8)); }

static inline uint32_t unicorn_counter(const uint8_t *frame)
{
    const uint8_t *p = frame + UNICORN_OFF_CNT;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline float unicorn_battery_pct(const uint8_t *frame)
{
    return 100.0f * (frame[UNICORN_OFF_BATT] & 0x0F) / 15.0f;
}
