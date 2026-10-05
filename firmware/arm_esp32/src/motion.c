#include "motion.h"

#include <math.h>
#include <stdio.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "motion";

// Límites de la etapa de pruebas (rad). Más estrechos que los del firmware del brazo.
static const pose_t LIM_MIN = {-1.6f, -0.6f, 0.6f, 1.6f};
static const pose_t LIM_MAX = { 1.6f,  0.9f, 2.6f, 3.2f};

#define STEPS_PER_RAD   (4096.0f / (2.0f * (float)M_PI))
#define STEP_ACC        100     // aceleración del servo en los pasos: alta para seguir de cerca
#define HOME_SPD        300     // pasos/s (~26 °/s)
#define HOME_ACC        10
#define HOME_MS         5000    // peor caso desde cualquier postura dentro de los límites
#define LINK_LOST_FAILS 4       // 4 pasos sin confirmar = 200 ms

static bool (*s_send)(const char *json);
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static motion_status_t s = {.state = MOTION_UNKNOWN, .vmax = 0.35f};  // ~20 °/s
static int64_t s_home_until_us;

const char *motion_state_name(motion_state_t st)
{
    switch (st) {
    case MOTION_UNKNOWN:   return "UNKNOWN";
    case MOTION_HOMING:    return "HOMING";
    case MOTION_IDLE:      return "IDLE";
    case MOTION_MOVING:    return "MOVING";
    case MOTION_LINK_LOST: return "LINK_LOST";
    }
    return "?";
}

static float approach(float from, float to, float max_step)
{
    float d = to - from;
    if (fabsf(d) <= max_step) return to;
    return from + copysignf(max_step, d);
}

static bool same(pose_t a, pose_t b)
{
    return a.b == b.b && a.s == b.s && a.e == b.e && a.h == b.h;
}

static void send_pose(pose_t p, int spd, int acc)
{
    char j[160];
    snprintf(j, sizeof j, "{\"T\":102,\"base\":%.4f,\"shoulder\":%.4f,\"elbow\":%.4f,\"hand\":%.4f,\"spd\":%d,\"acc\":%d}",
             p.b, p.s, p.e, p.h, spd, acc);
    s_send(j);
}

void motion_on_ack(bool ok)
{
    portENTER_CRITICAL(&s_mux);
    if (ok) {
        s.acks++;
        s.consec_fails = 0;
        if (s.state == MOTION_LINK_LOST) s.state = MOTION_IDLE;   // el brazo ya tiene la pose de espera
    } else {
        s.fails++;
        s.consec_fails++;
        if (s.state == MOTION_HOMING) {
            s.state = MOTION_UNKNOWN;
        } else if (s.consec_fails >= LINK_LOST_FAILS && s.state != MOTION_UNKNOWN) {
            s.state = MOTION_LINK_LOST;
            s.target = s.cmd;   // no seguir avanzando: el brazo no está recibiendo
        }
    }
    portEXIT_CRITICAL(&s_mux);
}

static void motion_task(void *arg)
{
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(MOTION_DT_MS));

        bool send = false;
        pose_t p;
        int spd = 0;
        motion_state_t prev, now;

        portENTER_CRITICAL(&s_mux);
        prev = s.state;
        if (s.frozen) {
            // simula supervisor muerto: no se envía nada
        } else if (s.state == MOTION_HOMING) {
            if (esp_timer_get_time() >= s_home_until_us) s.state = MOTION_IDLE;
        } else if (s.state == MOTION_LINK_LOST) {
            p = s.cmd;          // sonda: reenvía la pose de espera hasta que confirme
            send = true;
        } else if (s.state == MOTION_IDLE || s.state == MOTION_MOVING) {
            if (same(s.cmd, s.target)) {
                s.state = MOTION_IDLE;
            } else {
                float step = s.vmax * MOTION_DT_MS / 1000.0f;
                s.cmd.b = approach(s.cmd.b, s.target.b, step);
                s.cmd.s = approach(s.cmd.s, s.target.s, step);
                s.cmd.e = approach(s.cmd.e, s.target.e, step);
                s.cmd.h = approach(s.cmd.h, s.target.h, step);
                s.state = MOTION_MOVING;
                s.steps++;
                p = s.cmd;
                send = true;
            }
        }
        // Velocidad del servo con margen sobre vmax para que siga a los pasos sin quedarse atrás.
        spd = (int)lroundf(s.vmax * STEPS_PER_RAD * 1.5f);
        if (spd < 30) spd = 30;
        now = s.state;
        portEXIT_CRITICAL(&s_mux);

        if (send) send_pose(p, spd, STEP_ACC);
        if (now != prev) ESP_LOGI(TAG, "%s -> %s", motion_state_name(prev), motion_state_name(now));
    }
}

void motion_init(bool (*send_json)(const char *json))
{
    s_send = send_json;
    xTaskCreate(motion_task, "motion", 4096, NULL, 10, NULL);
}

void motion_home(void)
{
    pose_t p = POSE_INIT;
    portENTER_CRITICAL(&s_mux);
    s.state = MOTION_HOMING;
    s.cmd = s.target = p;
    s.consec_fails = 0;
    s_home_until_us = esp_timer_get_time() + (int64_t)HOME_MS * 1000;
    portEXIT_CRITICAL(&s_mux);
    ESP_LOGW(TAG, "home: movimiento lento a la postura inicial (%d pasos/s), %d ms", HOME_SPD, HOME_MS);
    send_pose(p, HOME_SPD, HOME_ACC);
}

bool motion_set_target(pose_t p, const char **why)
{
    motion_state_t st = motion_status().state;
    if (st != MOTION_IDLE && st != MOTION_MOVING) {
        *why = st == MOTION_LINK_LOST ? "enlace perdido" : st == MOTION_HOMING ? "home en curso" : "posición desconocida: usa home";
        return false;
    }
    if (p.b < LIM_MIN.b || p.b > LIM_MAX.b || p.s < LIM_MIN.s || p.s > LIM_MAX.s ||
        p.e < LIM_MIN.e || p.e > LIM_MAX.e || p.h < LIM_MIN.h || p.h > LIM_MAX.h) {
        *why = "fuera de los límites de prueba";
        return false;
    }
    bool ok = false;
    portENTER_CRITICAL(&s_mux);
    if (s.state == MOTION_IDLE || s.state == MOTION_MOVING) {
        s.target = p;
        ok = true;
    }
    portEXIT_CRITICAL(&s_mux);
    if (!ok) *why = "el estado cambió, reintenta";
    return ok;
}

void motion_stop(void)
{
    portENTER_CRITICAL(&s_mux);
    s.target = s.cmd;
    portEXIT_CRITICAL(&s_mux);
}

void motion_set_vmax(float rad_s)
{
    if (rad_s < 0.02f) rad_s = 0.02f;
    if (rad_s > 1.0f) rad_s = 1.0f;     // ~57 °/s en etapa de pruebas
    portENTER_CRITICAL(&s_mux);
    s.vmax = rad_s;
    portEXIT_CRITICAL(&s_mux);
}

void motion_freeze(bool on)
{
    portENTER_CRITICAL(&s_mux);
    s.frozen = on;
    portEXIT_CRITICAL(&s_mux);
}

void motion_invalidate(void)
{
    portENTER_CRITICAL(&s_mux);
    s.state = MOTION_UNKNOWN;
    portEXIT_CRITICAL(&s_mux);
}

motion_status_t motion_status(void)
{
    portENTER_CRITICAL(&s_mux);
    motion_status_t r = s;
    portEXIT_CRITICAL(&s_mux);
    return r;
}
