// Prueba en PC de firmware/common/arm_protocol.h:
//   gcc -Wall -Wextra -O2 -I../../common test_arm_protocol.c -o /tmp/t && /tmp/t
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "arm_protocol.h"

static int n_alert, n_event, n_status, n_other;
static arm_alert_t last_alert;

static void cb(uint8_t type, const uint8_t *p, uint8_t len, void *ctx)
{
    (void)ctx;
    arm_event_t e;
    arm_status_t s;
    switch (type) {
    case ARM_TYPE_ALERT:  if (arm_decode_alert(p, len, &last_alert)) n_alert++; else n_other++; break;
    case ARM_TYPE_EVENT:  assert(arm_decode_event(p, len, &e)); n_event++; break;
    case ARM_TYPE_STATUS: assert(arm_decode_status(p, len, &s)); n_status++; break;
    default: n_other++;
    }
}

int main(void)
{
    uint8_t f[ARM_MAX_FRAME], g[ARM_MAX_FRAME];
    arm_rx_t rx;

    // 1. Ida y vuelta de los tres tipos.
    arm_rx_init(&rx);
    arm_alert_t a = {.level = 2, .reason = ARM_REASON_ERRP, .seq = 65535, .t_ms = 123456, .score = 0.75f};
    size_t n = arm_encode_alert(f, &a);
    assert(n == 18);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_alert == 1 && last_alert.level == 2 && last_alert.seq == 65535 && last_alert.score == 0.75f);
    arm_event_t e = {.seq = 7, .action_id = 3, .kind = ARM_EVENT_ONSET, .label = ARM_LABEL_ERROR, .t_us = 99, .level = 0};
    arm_status_t s = {.safety = ARM_SAFETY_RUN, .level_rx = 0, .alerts_rx = 10};
    assert(arm_encode_event(f, &e) == 19);
    arm_rx_feed(&rx, f, 19, cb, NULL);
    assert(arm_encode_status(f, &s) == 26);
    arm_rx_feed(&rx, f, 26, cb, NULL);
    assert(n_event == 1 && n_status == 1);

    // 2. Basura antes, entre tramas y sync falsos (C3 sin 3C, C3 C3 3C).
    arm_rx_init(&rx);
    n_alert = 0;
    uint8_t junk[] = {0x00, 0xC3, 0x00, 0xC3, 0xC3, 0xFF, 0x3C, 0x12};
    n = arm_encode_alert(f, &a);
    arm_rx_feed(&rx, junk, sizeof junk, cb, NULL);
    arm_rx_feed(&rx, f, n, cb, NULL);
    arm_rx_feed(&rx, junk, 3, cb, NULL);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_alert == 2);

    // 3. CRC corrupto: se descarta, y la trama siguiente se recibe.
    arm_rx_init(&rx);
    n_alert = 0;
    memcpy(g, f, n);
    g[6] ^= 0x01;
    arm_rx_feed(&rx, g, n, cb, NULL);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_alert == 1 && rx.crc_errors == 1);

    // 4. len imposible: se descarta y resincroniza.
    arm_rx_init(&rx);
    n_alert = 0;
    uint8_t bad[] = {0xC3, 0x3C, 0x10, 200};
    arm_rx_feed(&rx, bad, sizeof bad, cb, NULL);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_alert == 1);

    // 5. Byte a byte con 10000 tramas mezcladas con ruido aleatorio que no contiene C3.
    arm_rx_init(&rx);
    n_alert = 0;
    srand(1);
    for (int i = 0; i < 10000; i++) {
        a.seq = (uint16_t)i;
        a.level = (uint8_t)(i % 4);
        n = arm_encode_alert(f, &a);
        for (size_t k = 0; k < n; k++) arm_rx_byte(&rx, f[k], cb, NULL);
        int r = rand() % 5;
        for (int k = 0; k < r; k++) { uint8_t z = (uint8_t)(rand() % 0xC3); arm_rx_byte(&rx, z, cb, NULL); }
    }
    assert(n_alert == 10000 && rx.crc_errors == 0);

    // 6. Nivel fuera de rango (CRC válido): arm_decode_alert lo rechaza.
    arm_rx_init(&rx);
    n_alert = 0; n_other = 0;
    a.level = 7;
    n = arm_encode_alert(f, &a);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_alert == 0 && n_other == 1);

    // 7. Tipo desconocido: se entrega al callback (que lo ignora), no rompe el receptor.
    arm_rx_init(&rx);
    n_other = 0; n_alert = 0;
    uint8_t pl[3] = {1, 2, 3};
    n = arm_encode(f, 0x77, pl, 3);
    arm_rx_feed(&rx, f, n, cb, NULL);
    a.level = 1;
    n = arm_encode_alert(f, &a);
    arm_rx_feed(&rx, f, n, cb, NULL);
    assert(n_other == 1 && n_alert == 1);

    printf("arm_protocol.h: todas las pruebas OK\n");
    return 0;
}
