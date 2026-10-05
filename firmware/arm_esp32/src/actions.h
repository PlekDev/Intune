// INTUNE C3: acciones del brazo con pulso de sync + EVENT (regla dura 4).
//
// Cada acción es un destino de motion.c. Justo antes de enviar su primer paso se sube el pulso
// de sync hacia el S3 (ARM_SYNC_PULSE_US) y el LED (ACTIONS_LED_MS, visible en video) y se envía
// un EVENT. Un movimiento que no es acción (retomar tras pausa, home) no genera pulso.
//
// Prueba de inicio ("onset"): repite una acción corta de la base N veces con intervalos
// aleatorios. Grabando LED + brazo en cámara lenta se mide el retraso pulso -> movimiento visible
// y su variación.
#pragma once
#include <stdint.h>

#define ACTIONS_LED_MS 150

void actions_init(void);
void actions_onset_test(int reps, float amp_deg);
void actions_stop(void);
void actions_led_test(void);   // 3 destellos: comprobar que la placa tiene LED en el GPIO configurado
