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
#include <stdbool.h>
#include <stdint.h>
#include "arm_protocol.h"
#include "motion.h"

#define ACTIONS_LED_MS        150
#define ACTIONS_MIN_SPACING_MS ARM_SYNC_MIN_SPACING_MS   // por debajo el S3 marca OVERLAP (1.5 s en grabación)

void actions_init(void);
// Acción con pulso: fija el destino y marca que su primer paso lleva pulso + EVENT.
// label: ARM_LABEL_* (solo para logs). Rechaza si el último pulso fue hace < ACTIONS_MIN_SPACING_MS.
bool actions_act(pose_t p, uint8_t label, const char **why);
void actions_disarm(void);     // un movimiento que no es acción (go, home, stop) cancela un pulso pendiente
void actions_onset_test(int reps, float amp_deg);
void actions_stop(void);
void actions_led_test(void);
void actions_sync_test(void);  // GPIO de sync en alto 3 s (para verlo con multímetro); nunca durante una acción   // 3 destellos: comprobar que la placa tiene LED en el GPIO configurado
