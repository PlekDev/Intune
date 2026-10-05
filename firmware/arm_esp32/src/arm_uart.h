// INTUNE C3: enlace UART supervisor -> RoArm-M2-S (firmware de fábrica, sin tocar).
//
// El brazo lee JSON por línea en su UART0 a 115200 (Serial, la misma del USB y del conector de
// 40 pines para Raspberry Pi). Con InfoPrint = 1 (valor de fábrica) repite cada línea JSON que
// entiende: ese eco es la confirmación de cada paso. {"T":105} responde con la posición
// ({"T":1051,...}), que el supervisor pide cada ARM_UART_FB_MS.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "motion.h"

#define ARM_UART_BAUD     115200
#define ARM_UART_ECHO_MS  150    // sin eco en este tiempo = paso no confirmado
#define ARM_UART_FB_MS    200    // periodo de lectura de posición

typedef struct {
    bool valid;
    pose_t pose;        // b, s, e, t del brazo (rad)
    int64_t age_ms;     // antigüedad de la última lectura
    uint32_t echoes, timeouts, fb_lines, other_lines;
    uint32_t rx_bytes;  // bytes en bruto recibidos (diagnóstico de cableado/baudios)
    float echo_ms_last, echo_ms_max;
} arm_uart_status_t;

void arm_uart_init(void);
// Envía una línea JSON. counts_for_motion: el eco (o su ausencia) llega a motion_on_ack().
bool arm_uart_send(const char *json, bool counts_for_motion, bool verbose);
arm_uart_status_t arm_uart_status(void);
// Diagnóstico de cableado: suelta el RX de la UART, muestrea el nivel del pin 1 s (con pull-down)
// y lo devuelve a la UART. Un TX en reposo está en alto; un pin suelto queda en bajo.
void arm_uart_probe(int pin);   // RX o TX del enlace; al terminar los devuelve a la UART
