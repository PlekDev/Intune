// INTUNE C3: movimiento por pasos cortos hacia el RoArm-M2-S.
//
// El brazo (firmware de fábrica) no tiene watchdog: termina la última orden que recibe.
// Por eso nunca se le manda un destino lejano: cada MOTION_DT_MS se le envía un punto que
// avanza como máximo vmax * dt desde el anterior. Si el supervisor o el enlace mueren, el
// brazo se detiene en el último punto recibido (a ~1 paso de donde estaba).
#pragma once
#include <stdbool.h>
#include <stdint.h>

#define MOTION_DT_MS 50

typedef struct {
    float b, s, e, h;   // base, hombro, codo, pinza (rad, convención de T:102)
} pose_t;

// Postura a la que va el brazo al arrancar (RoArmM2_moveInit).
#define POSE_INIT ((pose_t){0.0f, 0.0f, 1.5708f, 3.1416f})

typedef enum {
    MOTION_UNKNOWN,    // no se sabe dónde está el brazo: hace falta home
    MOTION_HOMING,     // movimiento lento a POSE_INIT (único destino lejano permitido)
    MOTION_IDLE,       // en el destino
    MOTION_MOVING,     // enviando pasos
    MOTION_LINK_LOST,  // el brazo dejó de confirmar: parado hasta que vuelva el enlace
} motion_state_t;

typedef struct {
    motion_state_t state;
    pose_t cmd, target;
    float vmax;               // rad/s (nominal)
    float speed_scale;        // 0..1, la fija la capa de seguridad (nivel 1)
    bool held;                // retenido por la capa de seguridad (niveles 2/3, sin S3)
    bool frozen;
    uint32_t steps, acks, fails, consec_fails;
} motion_status_t;

// send_json: envía un JSON al brazo; devuelve false si no pudo encolarlo.
void motion_init(bool (*send_json)(const char *json));
void motion_on_ack(bool ok);          // desde el callback de envío ESP-NOW
bool motion_home(const char **why);
bool motion_set_target(pose_t p, const char **why);
void motion_stop(void);               // destino = posición actual ordenada
void motion_set_vmax(float rad_s);
void motion_freeze(bool on);          // prueba: deja de transmitir de golpe
void motion_invalidate(void);         // alguien movió el brazo por fuera: hace falta home
// Capa de seguridad (safety.c):
void motion_set_speed_scale(float k);
// on: frena (destino = posición actual) y rechaza destinos nuevos. keep_target: guarda el destino
// para retomarlo al soltar (pausa de nivel 2); si no, se descarta (nivel 3, sin S3).
void motion_hold(bool on, bool keep_target);
// Se llama en la tarea de movimiento justo ANTES de enviar el primer paso de cada movimiento
// (IDLE -> MOVING). Es el instante del pulso de sync (regla dura 4). No debe bloquear.
void motion_set_onset_cb(void (*cb)(void));
motion_status_t motion_status(void);
const char *motion_state_name(motion_state_t s);
