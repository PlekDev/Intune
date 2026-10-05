// INTUNE: detector ErrP local en el supervisor (sin S3).
// Recibe del puente (C1) las tramas link_protocol.h por UART (RX = CONFIG_ARM_S3_UART_RX_GPIO, a
// LINK_BAUD), corta una época [-200, +800) ms en cada trama EVENT (el pulso de sync de este mismo
// supervisor llega al GPIO18 del puente, que lo convierte en número de muestra Unicorn), aplica la
// puerta de artefactos de CLAUDE.md, corre el ErrP-AE de C2 (autoencoder_engine.h, TFLite Micro
// int8) y entrega el nivel a safety.c por la misma ruta que el ALERT del S3 (heartbeat incluido).
// Si se pierde el EEG tras haberlo tenido, manda nivel 3 (ARM_REASON_EEG_LOST). Mientras la consola
// simula el S3 ("s3 <n>", p. ej. al grabar datos), el detector no manda nada.
#pragma once
#include <stdbool.h>

void detector_init(void);
void detector_log_status(void);
// El operador confirmó una pausa o rearmó tras una parada (CLAUDE.md: el nivel vuelve a 0). Sin EEG
// vivo no hace nada, así el rearme sigue fallando. Lo llaman safety_confirm() y safety_reset().
void detector_operator_ack(void);
