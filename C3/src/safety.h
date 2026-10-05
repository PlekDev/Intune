// INTUNE C3: capa de seguridad del supervisor.
// Recibe ALERT del S3 (arm_protocol.h) por UART, vigila el heartbeat, aplica la reacción de
// cada nivel sobre motion.c y atiende al operador (botón BOOT: corta = confirmar pausa,
// larga = rearmar tras parada). Envía STATUS al S3.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "arm_protocol.h"

void safety_init(void);
// Simulación del S3 desde la consola: misma ruta que la UART real (codifica y alimenta el
// parser). level < 0 detiene la simulación (= S3 muerto).
void safety_sim_s3(int level);
void safety_confirm(void);   // = pulsación corta
void safety_reset(void);     // = pulsación larga
uint8_t safety_state(void);  // ARM_SAFETY_*
const char *safety_state_name(uint8_t st);
void safety_log_status(void);
// El LED deja de mostrar el estado de seguridad mientras lo usa la prueba de inicio (actions.c).
void safety_led_override(bool on);
// Envía una trama EVENT al S3 (arm_protocol.h). Con el detector local no hay S3: no hace nada.
void safety_send_event(const arm_event_t *e);
// Detector local (detector.c): entrega un ALERT por la misma ruta que la UART del S3.
void safety_inject_alert(const arm_alert_t *a);
// true mientras la consola simula el S3 ("s3 <n>"): el detector local se calla.
bool safety_sim_active(void);
