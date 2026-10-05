// Puente Unicorn Hybrid Black (BT Classic SPP) -> UART -> ESP32-S3.
//
// Flujo BT (core 0): conectar (SDP si SCN=0) -> OPEN -> enviar STOP y descartar 300 ms ->
// enviar START -> primeros 3 bytes deben ser ACK 00 00 00 -> streaming al stream buffer.
// Desconexión / timeout / silencio -> reconexión con backoff.
//
// Proceso (core 1, proc_task): parsear tramas Unicorn -> huecos con ZOH + flags (nunca
// interpolar) -> IIR causal 1-15 Hz por canal (eeg_iir.h) -> tramas link_protocol.h por UART.
// Ver "Hard rules" en CLAUDE.md.

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_bt_device.h"
#include "esp_gap_bt_api.h"
#include "esp_spp_api.h"
#include "unicorn_protocol.h"
#include "unicorn_parser.h"
#include "link_protocol.h"
#include "eeg_iir.h"

static const char *TAG = "bridge";

#define BRIDGE_UART        UART_NUM_2
#define SB_SIZE            16384
#define UART_CHUNK         512
#define FLUSH_MS           300
#define ACK_TIMEOUT_MS     2000
#define CONNECT_TIMEOUT_MS 15000
#define BACKOFF_MIN_MS     1000
#define BACKOFF_MAX_MS     10000

#if CONFIG_BRIDGE_UNFILTERED
#define FILTER_NOTE " (SIN FILTRO)"
#else
#define FILTER_NOTE ""
#endif

#if CONFIG_UNICORN_SEC_AUTH
#define SPP_SEC ESP_SPP_SEC_AUTHENTICATE
#else
#define SPP_SEC ESP_SPP_SEC_NONE
#endif

typedef enum {
    ST_IDLE,        // desconectado, esperando reintento (o pausado)
    ST_INQUIRY,     // inquiry previo: obtiene clock offset / page scan mode del Unicorn
    ST_DISCOVERING, // SDP en curso
    ST_CONNECTING,  // esp_spp_connect en curso
    ST_FLUSH,       // conectado, STOP enviado, descartando datos
    ST_WAIT_ACK,    // START enviado, esperando 00 00 00
    ST_STREAMING,   // passthrough
} bridge_state_t;

static const char *STATE_NAMES[] = {"IDLE", "INQUIRY", "DISCOVERING", "CONNECTING", "FLUSH", "WAIT_ACK", "STREAMING"};

typedef enum {
    EV_INQ_DONE, EV_DISC_OK, EV_DISC_FAIL, EV_OPEN, EV_OPEN_FAIL, EV_CLOSE, EV_ACK_OK, EV_ACK_BAD,
} ev_type_t;

typedef struct {
    ev_type_t type;
    uint32_t arg;
} bridge_ev_t;

static volatile bridge_state_t s_state = ST_IDLE;
static volatile uint32_t s_handle;
static esp_bd_addr_t s_peer;
static uint8_t s_scn;
static QueueHandle_t s_evq;
static StreamBufferHandle_t s_sb;

// Estado del chequeo de ACK (solo lo toca el callback SPP)
static uint8_t s_ack_buf[3];
static int s_ack_idx;

// Stats
static volatile uint32_t s_rx_bytes, s_rx_flushed, s_sb_drops, s_uart_bytes;
static volatile uint32_t s_reconnects, s_cong_events, s_sessions;
static volatile int64_t s_last_rx_us;
static volatile bool s_congested;
static volatile bool s_paused;

// Inquiry (solo lo tocan gap_cb y ctrl_task, secuencialmente)
static volatile bool s_inq_found;
static volatile int s_inq_rssi;
static volatile int s_inq_devices;

// Proceso: solo lo toca proc_task (core 1); stats lee los contadores
static unicorn_parser_t s_parser;
static volatile bool s_parser_new_session; // ctrl -> proc_task: el contador reinicia en 1
static volatile bool s_parser_dump_next;   // imprimir la próxima trama decodificada
static unicorn_sample_t s_last_sample;

static eeg_iir_t s_iir[UNICORN_N_EEG];
static float s_hold_uv[UNICORN_N_EEG];     // última entrada real, para ZOH
static int16_t s_hold_imu[6];              // acc[3] + gyr[3] crudos de esa muestra
static bool s_need_reset = true;
static bool s_session_start = true;
static uint32_t s_settle_left;
static uint32_t s_seen_backwards;
static uint32_t s_proc_us_max;             // máximo del segundo en curso
static volatile uint32_t s_proc_us_last;   // máximo del último segundo (STATUS y stats)
static volatile uint32_t s_held, s_filter_resets, s_link_frames;

// Pulso de sincronía: t0 = envolvente inferior de (t_llegada - contador*4 ms), mínimo
// de los últimos T0_BLOCKS bloques de T0_BLOCK_LEN muestras (sigue la deriva de reloj).
#define T0_BLOCK_LEN   500  // 2 s
#define T0_BLOCKS      5    // ventana de 10 s
#define T0_MIN_SAMPLES 250  // 1 s de datos antes de dar t0 por válido
#define SYNC_DEBOUNCE_US 20000
#define SYNC_OVERLAP_US  1000000
static QueueHandle_t s_sync_q;             // ISR -> proc_task: int64 t_flanco (us)
static int64_t s_t0_blk[T0_BLOCKS];
static int s_t0_blk_idx, s_t0_blk_fill;
static uint32_t s_t0_samples;
static volatile uint32_t s_events;

static void t0_reset(void)
{
    for (int i = 0; i < T0_BLOCKS; i++) {
        s_t0_blk[i] = INT64_MAX;
    }
    s_t0_blk_idx = s_t0_blk_fill = 0;
    s_t0_samples = 0;
}

static void t0_update(int64_t t_arr, uint32_t counter)
{
    int64_t c = t_arr - (int64_t)counter * 4000;
    if (c < s_t0_blk[s_t0_blk_idx]) {
        s_t0_blk[s_t0_blk_idx] = c;
    }
    s_t0_samples++;
    if (++s_t0_blk_fill >= T0_BLOCK_LEN) {
        s_t0_blk_fill = 0;
        s_t0_blk_idx = (s_t0_blk_idx + 1) % T0_BLOCKS;
        s_t0_blk[s_t0_blk_idx] = INT64_MAX;
    }
}

static bool t0_get(int64_t *t0)
{
    if (s_t0_samples < T0_MIN_SAMPLES) {
        return false;
    }
    int64_t m = INT64_MAX;
    for (int i = 0; i < T0_BLOCKS; i++) {
        if (s_t0_blk[i] < m) {
            m = s_t0_blk[i];
        }
    }
    *t0 = m;
    return true;
}

#if CONFIG_BRIDGE_SYNC_GPIO >= 0
static void IRAM_ATTR sync_isr(void *arg)
{
    int64_t t = esp_timer_get_time(); // nada más: el resto en proc_task
    BaseType_t hp = pdFALSE;
    xQueueSendFromISR(s_sync_q, &t, &hp);
    if (hp) {
        portYIELD_FROM_ISR();
    }
}
#endif

static void link_write(const uint8_t *f, size_t n)
{
    uart_write_bytes(BRIDGE_UART, f, n);
#if CONFIG_BRIDGE_LINK_MIRROR_CONSOLE
    uart_write_bytes(UART_NUM_0, f, n);
#endif
    s_uart_bytes += n;
}

// Filtra y envía una muestra, real o retenida (ZOH).
static void emit_sample(uint32_t counter, const float uv[UNICORN_N_EEG], const int16_t imu[6], uint8_t flags)
{
    if (s_settle_left) {
        flags |= LINK_F_SETTLING;
        s_settle_left--;
    }
#if CONFIG_BRIDGE_UNFILTERED
    flags |= LINK_F_UNFILTERED;
#endif
    link_eeg_t s = {.counter = counter, .flags = flags};
    memcpy(s.acc, imu, sizeof(s.acc));
    memcpy(s.gyr, imu + 3, sizeof(s.gyr));
    for (int ch = 0; ch < UNICORN_N_EEG; ch++) {
        double y = eeg_iir_step(&s_iir[ch], uv[ch]); // el estado avanza también en UNFILTERED
#if CONFIG_BRIDGE_UNFILTERED
        s.eeg_uv[ch] = uv[ch];
#else
        s.eeg_uv[ch] = (float)y;
#endif
    }
    uint8_t f[LINK_MAX_FRAME];
    link_write(f, link_encode_eeg(f, &s));
#if CONFIG_BRIDGE_LINK_RAW
    memcpy(s.eeg_uv, uv, sizeof(s.eeg_uv));
    link_write(f, link_encode_eeg_raw(f, &s));
#endif
    s_link_frames++;
}

static void filters_reset(const float uv[UNICORN_N_EEG])
{
    for (int ch = 0; ch < UNICORN_N_EEG; ch++) {
        eeg_iir_reset(&s_iir[ch], uv[ch]);
    }
    s_settle_left = EEG_IIR_SETTLE_SAMPLES;
    s_filter_resets++;
}

static void on_frame(const uint8_t *f, uint32_t gap, void *ctx)
{
    int64_t t_arr = esp_timer_get_time(); // llegada (aprox.) para t0; también mide el proceso
    unicorn_decode(f, &s_last_sample);
    const unicorn_sample_t *m = &s_last_sample;
    uint8_t flags = 0;
    int16_t imu[6];
    for (int i = 0; i < 6; i++) {
        imu[i] = unicorn_i16le(f + UNICORN_OFF_ACC + 2 * i); // acc y gyr son contiguos (27..38)
    }

    if (s_parser.backwards != s_seen_backwards) {
        // Contador reiniciado sin pasar por OPEN (headset reiniciado): sesión nueva
        s_seen_backwards = s_parser.backwards;
        s_need_reset = s_session_start = true;
    }
    if (gap) {
        ESP_LOGW(TAG, "[valid] hueco: %" PRIu32 " muestras antes de cnt=%" PRIu32, gap, m->counter);
    }

    if (s_need_reset) {
        s_need_reset = false;
        if (s_session_start) {
            flags |= LINK_F_SESSION_START;
            s_session_start = false;
        }
        filters_reset(m->eeg_uv);
        flags |= LINK_F_FILTER_RESET;
    } else if (gap > LINK_MAX_HOLD) {
        // Hueco largo: no se rellena; el contador salta y el filtro arranca de nuevo
        filters_reset(m->eeg_uv);
        flags |= LINK_F_FILTER_RESET;
    } else if (gap) {
        // Regla dura 3: retención de orden cero, nunca interpolación
        uint8_t hf = LINK_F_HELD | (gap >= 2 ? LINK_F_GAP : 0);
        for (uint32_t k = gap; k >= 1; k--) {
            emit_sample(m->counter - k, s_hold_uv, s_hold_imu, hf);
        }
        s_held += gap;
    }
    emit_sample(m->counter, m->eeg_uv, imu, flags);
    memcpy(s_hold_uv, m->eeg_uv, sizeof(s_hold_uv));
    if (flags & LINK_F_SESSION_START) {
        t0_reset(); // el contador volvió a 1: t0 cambia por completo
    }
    t0_update(t_arr, m->counter);
    memcpy(s_hold_imu, imu, sizeof(s_hold_imu));

    if (s_parser_dump_next) {
        s_parser_dump_next = false;
        ESP_LOGI(TAG, "[valid] trama cnt=%" PRIu32 " bat=%.0f%% eeg=[%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f] uV "
                 "acc=[%.2f %.2f %.2f] g gyr=[%.1f %.1f %.1f] dps",
                 m->counter, m->battery_pct, m->eeg_uv[0], m->eeg_uv[1], m->eeg_uv[2], m->eeg_uv[3],
                 m->eeg_uv[4], m->eeg_uv[5], m->eeg_uv[6], m->eeg_uv[7], m->acc_g[0], m->acc_g[1], m->acc_g[2],
                 m->gyr_dps[0], m->gyr_dps[1], m->gyr_dps[2]);
    }

    uint32_t dt = (uint32_t)(esp_timer_get_time() - t_arr);
    if (dt > s_proc_us_max) {
        s_proc_us_max = dt;
    }
}

static void send_status(void)
{
    uint8_t state = LINK_STATE_CONNECTING;
    if (s_state == ST_STREAMING) {
        state = LINK_STATE_STREAMING;
    } else if (s_paused) {
        state = LINK_STATE_IDLE;
    }
    s_proc_us_last = s_proc_us_max;
    s_proc_us_max = 0;
    link_status_t st = {
        .state = state,
        .battery_pct = s_parser.frames ? (uint8_t)(s_last_sample.battery_pct + 0.5f) : 0xFF,
        .proc_us_max = s_proc_us_last > 0xFFFF ? 0xFFFF : (uint16_t)s_proc_us_last,
        .frames = s_parser.frames,
        .gaps = s_parser.gaps,
        .lost = s_parser.lost,
        .reconnects = s_reconnects,
    };
    uint8_t f[LINK_MAX_FRAME];
    link_write(f, link_encode_status(f, &st));
}

static void post(ev_type_t t, uint32_t arg)
{
    bridge_ev_t ev = {t, arg};
    xQueueSend(s_evq, &ev, 0); // nunca bloquear en callbacks BT
}

static void set_state(bridge_state_t st)
{
    if (s_state != st) {
        ESP_LOGI(TAG, "estado %s -> %s", STATE_NAMES[s_state], STATE_NAMES[st]);
        s_state = st;
    }
}

static void spp_send(const uint8_t *cmd, int len)
{
    if (s_congested) {
        ESP_LOGW(TAG, "SPP congestionado, envío igualmente");
    }
    esp_err_t err = esp_spp_write(s_handle, len, (uint8_t *)cmd);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_spp_write: %s", esp_err_to_name(err));
    }
}

// ---------------- Callbacks Bluetooth (rápidos, sin bloqueo) ----------------

static void on_data(const uint8_t *data, uint16_t len)
{
    s_last_rx_us = esp_timer_get_time();
    s_rx_bytes += len;

    switch (s_state) {
    case ST_FLUSH:
        s_rx_flushed += len;
        return;
    case ST_WAIT_ACK: {
        uint16_t i = 0;
        while (i < len && s_ack_idx < 3) {
            s_ack_buf[s_ack_idx++] = data[i++];
        }
        if (s_ack_idx < 3) {
            return;
        }
        bool ok = memcmp(s_ack_buf, UNICORN_ACK, 3) == 0;
        // Pasar a streaming aquí mismo para no perder lo que viene detrás del ACK
        s_state = ST_STREAMING;
        post(ok ? EV_ACK_OK : EV_ACK_BAD, (s_ack_buf[0] << 16) | (s_ack_buf[1] << 8) | s_ack_buf[2]);
        if (!ok) {
            // No era ACK: pasarlos al parser también, él resincroniza
            if (xStreamBufferSend(s_sb, s_ack_buf, 3, 0) != 3) {
                s_sb_drops += 3;
            }
        }
        data += i;
        len -= i;
        if (len == 0) {
            return;
        }
    } /* fallthrough */
    case ST_STREAMING: {
        size_t sent = xStreamBufferSend(s_sb, data, len, 0);
        if (sent != len) {
            s_sb_drops += len - sent;
        }
        return;
    }
    default:
        return;
    }
}

static void spp_cb(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
{
    switch (event) {
    case ESP_SPP_INIT_EVT:
        ESP_LOGI(TAG, "SPP init status=%d", param->init.status);
        break;
    case ESP_SPP_DISCOVERY_COMP_EVT:
        ESP_LOGI(TAG, "SDP status=%d scn_num=%d", param->disc_comp.status, param->disc_comp.scn_num);
        for (int i = 0; i < param->disc_comp.scn_num; i++) {
            ESP_LOGI(TAG, "  scn[%d]=%d %s", i, param->disc_comp.scn[i],
                     param->disc_comp.service_name[i] ? param->disc_comp.service_name[i] : "");
        }
        if (param->disc_comp.status == ESP_SPP_SUCCESS && param->disc_comp.scn_num > 0) {
            post(EV_DISC_OK, param->disc_comp.scn[0]);
        } else {
            post(EV_DISC_FAIL, param->disc_comp.status);
        }
        break;
    case ESP_SPP_CL_INIT_EVT:
        ESP_LOGD(TAG, "CL_INIT status=%d", param->cl_init.status);
        if (param->cl_init.status != ESP_SPP_SUCCESS) {
            post(EV_OPEN_FAIL, param->cl_init.status);
        }
        break;
    case ESP_SPP_OPEN_EVT:
        if (param->open.status == ESP_SPP_SUCCESS) {
            post(EV_OPEN, param->open.handle);
        } else {
            post(EV_OPEN_FAIL, param->open.status);
        }
        break;
    case ESP_SPP_CLOSE_EVT:
        post(EV_CLOSE, param->close.status);
        break;
    case ESP_SPP_DATA_IND_EVT:
        on_data(param->data_ind.data, param->data_ind.len);
        break;
    case ESP_SPP_CONG_EVT:
        s_congested = param->cong.cong;
        if (param->cong.cong) {
            s_cong_events++;
        }
        break;
    case ESP_SPP_WRITE_EVT:
        s_congested = param->write.cong;
        ESP_LOGD(TAG, "WRITE status=%d len=%d", param->write.status, param->write.len);
        break;
    default:
        ESP_LOGD(TAG, "SPP evt %d", event);
        break;
    }
}

static void gap_cb(esp_bt_gap_cb_event_t event, esp_bt_gap_cb_param_t *param)
{
    switch (event) {
    case ESP_BT_GAP_DISC_RES_EVT: {
        s_inq_devices++;
        for (int i = 0; i < param->disc_res.num_prop; i++) {
            if (param->disc_res.prop[i].type == ESP_BT_GAP_DEV_PROP_RSSI) {
                ESP_LOGI(TAG, "  inquiry: " ESP_BD_ADDR_STR " RSSI=%d", ESP_BD_ADDR_HEX(param->disc_res.bda),
                         *(int8_t *)param->disc_res.prop[i].val);
            }
        }
        if (memcmp(param->disc_res.bda, s_peer, ESP_BD_ADDR_LEN) == 0 && !s_inq_found) {
            s_inq_found = true;
            for (int i = 0; i < param->disc_res.num_prop; i++) {
                if (param->disc_res.prop[i].type == ESP_BT_GAP_DEV_PROP_RSSI) {
                    s_inq_rssi = *(int8_t *)param->disc_res.prop[i].val;
                }
            }
            esp_bt_gap_cancel_discovery(); // -> DISC_STATE_CHANGED(STOPPED)
        }
        break;
    }
    case ESP_BT_GAP_DISC_STATE_CHANGED_EVT:
        if (param->disc_st_chg.state == ESP_BT_GAP_DISCOVERY_STOPPED && s_state == ST_INQUIRY) {
            post(EV_INQ_DONE, s_inq_found);
        }
        break;
    case ESP_BT_GAP_AUTH_CMPL_EVT:
        ESP_LOGI(TAG, "AUTH_CMPL status=%d (%s)", param->auth_cmpl.stat,
                 param->auth_cmpl.stat == ESP_BT_STATUS_SUCCESS ? "OK" : "FALLO");
        break;
    case ESP_BT_GAP_PIN_REQ_EVT: {
        // Legacy pairing (no observado en Linux, por si acaso)
        ESP_LOGW(TAG, "PIN_REQ (legacy pairing), respondiendo '%s'", CONFIG_UNICORN_PIN);
        esp_bt_pin_code_t pin = {0};
        int n = strlen(CONFIG_UNICORN_PIN);
        memcpy(pin, CONFIG_UNICORN_PIN, n > 16 ? 16 : n);
        esp_bt_gap_pin_reply(param->pin_req.bda, true, n, pin);
        break;
    }
    case ESP_BT_GAP_CFM_REQ_EVT:
        ESP_LOGW(TAG, "SSP CFM_REQ num=%" PRIu32 ", aceptando", param->cfm_req.num_val);
        esp_bt_gap_ssp_confirm_reply(param->cfm_req.bda, true);
        break;
    case ESP_BT_GAP_KEY_NOTIF_EVT:
        ESP_LOGW(TAG, "SSP KEY_NOTIF passkey=%" PRIu32, param->key_notif.passkey);
        break;
    case ESP_BT_GAP_SET_PAGE_TO_EVT:
        ESP_LOGI(TAG, "page timeout fijado (status=%d)", param->set_page_timeout.stat);
        break;
    case ESP_BT_GAP_MODE_CHG_EVT:
        ESP_LOGD(TAG, "MODE_CHG mode=%d", param->mode_chg.mode);
        break;
    default:
        ESP_LOGD(TAG, "GAP evt %d", event);
        break;
    }
}

// ---------------- Tareas ----------------

static void handle_edge(int64_t te)
{
    static int64_t last_edge = INT64_MIN / 2;
    static uint32_t seq;
    if (te - last_edge < SYNC_DEBOUNCE_US) {
        return; // rebote del flanco
    }
    link_event_t e = {.seq = ++seq};
    if (te - last_edge < SYNC_OVERLAP_US) {
        e.flags |= LINK_EVT_F_OVERLAP;
    }
    last_edge = te;
    int64_t t0;
    if (s_state == ST_STREAMING && t0_get(&t0) && te > t0) {
        int64_t d = te - t0;
        int64_t k = (d + 2000) / 4000;
        e.counter = (uint32_t)k;
        e.offset_us = (int16_t)(d - k * 4000);
    } else {
        e.flags |= LINK_EVT_F_NO_T0;
    }
    uint8_t f[LINK_MAX_FRAME];
    link_write(f, link_encode_event(f, &e));
    s_events++;
    ESP_LOGI(TAG, "[sync] flanco %" PRIu32 " -> cnt=%" PRIu32 " off=%d us flags=%02x", e.seq, e.counter,
             e.offset_us, e.flags);
}

// Core 1: stream buffer -> parser -> huecos + IIR -> UART. STATUS cada segundo.
static void proc_task(void *arg)
{
    static uint8_t buf[UART_CHUNK];
    int64_t next_status = esp_timer_get_time();
    for (;;) {
        size_t n = xStreamBufferReceive(s_sb, buf, sizeof(buf), pdMS_TO_TICKS(20));
        if (s_parser_new_session) {
            s_parser_new_session = false;
            unicorn_parser_new_session(&s_parser);
            s_need_reset = s_session_start = true;
        }
        if (n > 0) {
            unicorn_parser_feed(&s_parser, buf, n, on_frame, NULL);
        }
        int64_t te;
        while (xQueueReceive(s_sync_q, &te, 0) == pdTRUE) {
            handle_edge(te);
        }
        int64_t now = esp_timer_get_time();
        if (now >= next_status) {
            send_status();
            next_status = now + 1000000;
        }
    }
}

// LED: apagado = desconectado, parpadeo lento = conectando,
// parpadeo rápido = conectado sin streaming, fijo = streaming. GPIO estado = streaming.
static void led_task(void *arg)
{
    uint32_t tick = 0;
    for (;;) {
        bridge_state_t st = s_state;
        int on;
        switch (st) {
        case ST_INQUIRY:
        case ST_DISCOVERING:
        case ST_CONNECTING: on = (tick / 10) % 2; break;  // 1 Hz
        case ST_FLUSH:
        case ST_WAIT_ACK:   on = (tick / 2) % 2; break;   // 5 Hz
        case ST_STREAMING:  on = 1; break;
        default:            on = 0; break;
        }
#if CONFIG_BRIDGE_LED_GPIO >= 0
        gpio_set_level(CONFIG_BRIDGE_LED_GPIO, on);
#endif
        gpio_set_level(CONFIG_BRIDGE_STATUS_GPIO, st == ST_STREAMING);
        tick++;
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

static void start_connect(void)
{
    if (s_scn == 0) {
        set_state(ST_DISCOVERING);
        esp_err_t err = esp_spp_start_discovery(s_peer);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "start_discovery: %s", esp_err_to_name(err));
            post(EV_DISC_FAIL, err);
        }
    } else {
        set_state(ST_CONNECTING);
        ESP_LOGI(TAG, "conectando scn=%d sec=%s", s_scn, SPP_SEC == ESP_SPP_SEC_NONE ? "NONE" : "AUTH");
        esp_err_t err = esp_spp_connect(SPP_SEC, ESP_SPP_ROLE_MASTER, s_scn, s_peer);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "spp_connect: %s", esp_err_to_name(err));
            post(EV_OPEN_FAIL, err);
        }
    }
}

// Inquiry antes de conectar: sin él, el ESP32 llama al Unicorn sin clock offset ni
// page scan mode y obtiene Page Timeout (0x4) aunque la PC (que sí hace inquiry) conecte.
static void start_attempt(void)
{
#if CONFIG_UNICORN_INQUIRY_FIRST
    s_inq_found = false;
    s_inq_rssi = 0;
    s_inq_devices = 0;
    set_state(ST_INQUIRY);
    esp_err_t err = esp_bt_gap_start_discovery(ESP_BT_INQ_MODE_GENERAL_INQUIRY, 8, 0); // 8 × 1.28 s máx
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "start_discovery (inquiry): %s", esp_err_to_name(err));
        post(EV_INQ_DONE, 0);
    }
#else
    start_connect();
#endif
}

static bool button_pressed(void)
{
#if CONFIG_BRIDGE_BUTTON_GPIO >= 0
    static int prev = 1;
    int lvl = gpio_get_level(CONFIG_BRIDGE_BUTTON_GPIO);
    bool edge = (prev == 1 && lvl == 0);
    prev = lvl;
    return edge;
#else
    return false;
#endif
}

static void ctrl_task(void *arg)
{
    uint32_t backoff = BACKOFF_MIN_MS;
    int64_t retry_at = 0;             // us; 0 = sin reintento programado
    int64_t deadline = 0;             // us; timeout del estado actual
    int64_t last_stats = esp_timer_get_time();
    uint32_t last_rx = 0;
    bool connected = false;           // hay handle SPP abierto
    uint8_t scn_cfg = CONFIG_UNICORN_SCN;

    s_scn = scn_cfg;
    retry_at = esp_timer_get_time();  // conectar ya

    for (;;) {
        bridge_ev_t ev;
        int64_t now;
        if (xQueueReceive(s_evq, &ev, pdMS_TO_TICKS(50)) == pdTRUE) {
            now = esp_timer_get_time();
            switch (ev.type) {
            case EV_INQ_DONE:
                if (s_state == ST_INQUIRY) {
                    if (ev.arg) {
                        ESP_LOGI(TAG, "inquiry: Unicorn visto, RSSI=%d dBm (%d dispositivos)", s_inq_rssi, s_inq_devices);
                    } else {
                        ESP_LOGW(TAG, "inquiry: Unicorn NO visto (%d dispositivos); intento conectar igual", s_inq_devices);
                    }
                    start_connect();
                    deadline = now + CONNECT_TIMEOUT_MS * 1000LL;
                }
                break;
            case EV_DISC_OK:
                if (s_state == ST_DISCOVERING) {
                    s_scn = ev.arg;
                    start_connect();
                    deadline = now + CONNECT_TIMEOUT_MS * 1000LL;
                }
                break;
            case EV_DISC_FAIL:
            case EV_OPEN_FAIL:
                ESP_LOGW(TAG, "%s falló (status=%" PRIu32 ")", ev.type == EV_DISC_FAIL ? "SDP" : "conexión", ev.arg);
                if (!connected) {
                    set_state(ST_IDLE);
                    deadline = 0;
                    retry_at = s_paused ? 0 : now + backoff * 1000LL;
                    ESP_LOGI(TAG, "reintento en %" PRIu32 " ms", backoff);
                    backoff = backoff * 2 > BACKOFF_MAX_MS ? BACKOFF_MAX_MS : backoff * 2;
                }
                break;
            case EV_OPEN:
                connected = true;
                s_handle = ev.arg;
                s_sessions++;
                ESP_LOGI(TAG, "SPP abierto handle=%" PRIu32 " scn=%d", ev.arg, s_scn);
                set_state(ST_FLUSH);
                s_parser_new_session = true;     // el contador del Unicorn reinicia en 1
                spp_send(UNICORN_CMD_STOP, 3);   // por si quedó transmitiendo
                deadline = now + FLUSH_MS * 1000LL;
                break;
            case EV_ACK_OK:
                ESP_LOGI(TAG, "ACK OK -> streaming");
                s_parser_dump_next = true;
                backoff = BACKOFF_MIN_MS;
                deadline = 0;
                s_last_rx_us = now;
                set_state(ST_STREAMING); // el callback ya lo puso; esto solo loggea
                break;
            case EV_ACK_BAD:
                ESP_LOGW(TAG, "primeros 3 bytes tras START = %06" PRIx32 " (no es ACK); sigo en passthrough", ev.arg);
                deadline = 0;
                s_last_rx_us = now;
                set_state(ST_STREAMING);
                break;
            case EV_CLOSE:
                ESP_LOGW(TAG, "SPP cerrado (status=%" PRIu32 ")", ev.arg);
                if (connected || s_state == ST_CONNECTING) {
                    s_reconnects++;
                }
                connected = false;
                set_state(ST_IDLE);
                deadline = 0;
                retry_at = s_paused ? 0 : now + backoff * 1000LL;
                if (!s_paused) {
                    ESP_LOGI(TAG, "reintento en %" PRIu32 " ms", backoff);
                }
                backoff = backoff * 2 > BACKOFF_MAX_MS ? BACKOFF_MAX_MS : backoff * 2;
                break;
            }
        }
        now = esp_timer_get_time();

        // Reintento programado
        if (s_state == ST_IDLE && retry_at && now >= retry_at && !s_paused) {
            retry_at = 0;
            s_scn = scn_cfg;
            start_attempt();
            deadline = now + CONNECT_TIMEOUT_MS * 1000LL;
        }

        // Fin del flush -> START
        if (s_state == ST_FLUSH && now >= deadline) {
            ESP_LOGI(TAG, "descartados %" PRIu32 " B tras STOP; enviando START", s_rx_flushed);
            s_rx_flushed = 0;
            s_ack_idx = 0;
            set_state(ST_WAIT_ACK);
            spp_send(UNICORN_CMD_START, 3);
            deadline = now + ACK_TIMEOUT_MS * 1000LL;
        }

        // Timeouts de conexión / ACK
        if (deadline && now >= deadline &&
            (s_state == ST_INQUIRY || s_state == ST_DISCOVERING || s_state == ST_CONNECTING ||
             s_state == ST_WAIT_ACK)) {
            ESP_LOGW(TAG, "timeout en %s", STATE_NAMES[s_state]);
            deadline = 0;
            if (s_state == ST_INQUIRY) {
                esp_bt_gap_cancel_discovery();
            }
            if (connected) {
                esp_spp_disconnect(s_handle); // -> EV_CLOSE -> reintento
            } else {
                set_state(ST_IDLE);
                retry_at = now + backoff * 1000LL;
            }
        }

        // Silencio en streaming (headset apagado, fuera de alcance...)
        if (s_state == ST_STREAMING && now - s_last_rx_us > CONFIG_BRIDGE_RX_TIMEOUT_MS * 1000LL) {
            ESP_LOGW(TAG, "sin datos %d ms, desconectando", CONFIG_BRIDGE_RX_TIMEOUT_MS);
            s_last_rx_us = now;
            esp_spp_disconnect(s_handle);
        }

        // Botón: desconexión ordenada (STOP) / reanudar
        if (button_pressed()) {
            if (!s_paused) {
                s_paused = true;
                retry_at = 0;
                ESP_LOGI(TAG, "botón: STOP y desconexión ordenada");
                if (connected) {
                    spp_send(UNICORN_CMD_STOP, 3);
                    vTaskDelay(pdMS_TO_TICKS(200));
                    esp_spp_disconnect(s_handle);
                } else {
                    set_state(ST_IDLE);
                }
            } else {
                s_paused = false;
                ESP_LOGI(TAG, "botón: reanudar");
                if (s_state == ST_IDLE) {
                    retry_at = now;
                }
            }
        }

        // Stats
        if (now - last_stats >= CONFIG_BRIDGE_STATS_PERIOD_S * 1000000LL) {
            uint32_t rx = s_rx_bytes;
            float dt = (now - last_stats) / 1e6f;
            ESP_LOGI(TAG, "[stats] %s rx=%" PRIu32 " B (%.2f kB/s ~%.0f tramas/s) uart=%" PRIu32
                     " drops=%" PRIu32 " sb_libre=%u cong=%" PRIu32 " sesiones=%" PRIu32 " reconex=%" PRIu32,
                     STATE_NAMES[s_state], rx, (rx - last_rx) / dt / 1000.0f,
                     (rx - last_rx) / dt / UNICORN_FRAME_LEN, s_uart_bytes, s_sb_drops,
                     (unsigned)xStreamBufferSpacesAvailable(s_sb), s_cong_events, s_sessions, s_reconnects);
            {
                static uint32_t last_frames;
                const unicorn_parser_t *p = &s_parser;
                uint32_t fr = p->frames;
                ESP_LOGI(TAG, "[valid] tramas=%" PRIu32 " (%.1f/s) cnt=%" PRIu32 " huecos=%" PRIu32 " perdidas=%" PRIu32
                         " (%.3f%%) corruptas=%" PRIu32 " reinicios_cnt=%" PRIu32 " descartados=%" PRIu32 " B bat=%.0f%%",
                         fr, (fr - last_frames) / dt, s_last_sample.counter, p->gaps, p->lost,
                         unicorn_parser_loss_pct(p), p->corrupt, p->backwards, p->discarded, s_last_sample.battery_pct);
                last_frames = fr;
                s_parser_dump_next = true; // una trama decodificada por periodo
            }
            ESP_LOGI(TAG, "[iir] link=%" PRIu32 " retenidas(ZOH)=%" PRIu32 " reinicios_filtro=%" PRIu32
                     " proc_max=%" PRIu32 " us/muestra eventos=%" PRIu32 "%s",
                     s_link_frames, s_held, s_filter_resets, s_proc_us_last, s_events, FILTER_NOTE);
            last_rx = rx;
            last_stats = now;
        }
    }
}

// ---------------- Init ----------------

static void init_gpio_uart(void)
{
    uint64_t out_mask = 1ULL << CONFIG_BRIDGE_STATUS_GPIO;
#if CONFIG_BRIDGE_LED_GPIO >= 0
    out_mask |= 1ULL << CONFIG_BRIDGE_LED_GPIO;
#endif
    gpio_config_t out = {.pin_bit_mask = out_mask, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&out));
    gpio_set_level(CONFIG_BRIDGE_STATUS_GPIO, 0);
#if CONFIG_BRIDGE_BUTTON_GPIO >= 0
    gpio_config_t in = {.pin_bit_mask = 1ULL << CONFIG_BRIDGE_BUTTON_GPIO, .mode = GPIO_MODE_INPUT,
                        .pull_up_en = GPIO_PULLUP_ENABLE};
    ESP_ERROR_CHECK(gpio_config(&in));
#endif

#if CONFIG_BRIDGE_SYNC_GPIO >= 0
    gpio_config_t sync = {.pin_bit_mask = 1ULL << CONFIG_BRIDGE_SYNC_GPIO, .mode = GPIO_MODE_INPUT,
                          .pull_down_en = GPIO_PULLDOWN_ENABLE, .intr_type = GPIO_INTR_POSEDGE};
    ESP_ERROR_CHECK(gpio_config(&sync));
    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(CONFIG_BRIDGE_SYNC_GPIO, sync_isr, NULL));
#endif

    uart_config_t uc = {
        .baud_rate = CONFIG_BRIDGE_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(BRIDGE_UART, 256, 8192, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(BRIDGE_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(BRIDGE_UART, CONFIG_BRIDGE_UART_TX_GPIO, CONFIG_BRIDGE_UART_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

#if CONFIG_BRIDGE_LINK_MIRROR_CONSOLE
    // UART0 deja de ser consola: solo tramas link (los logs las romperían)
    ESP_LOGW(TAG, "logs silenciados; UART0 = tramas link a %d baud (tools/linux_probe/link_view.py)",
             CONFIG_BRIDGE_UART_BAUD);
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(20));
    esp_log_level_set("*", ESP_LOG_NONE);
    esp_log_level_set(TAG, ESP_LOG_NONE);
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 256, 8192, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uc));
#endif
}

static void init_bt(void)
{
    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_BLE));
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_CLASSIC_BT));
    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    ESP_ERROR_CHECK(esp_bt_gap_register_callback(gap_cb));
    ESP_ERROR_CHECK(esp_spp_register_callback(spp_cb));
    esp_spp_cfg_t spp_cfg = {
        .mode = ESP_SPP_MODE_CB,
        .enable_l2cap_ertm = true,
        .tx_buffer_size = 0,
    };
    ESP_ERROR_CHECK(esp_spp_enhanced_init(&spp_cfg));

    // Seguridad: SSP "Just Works" y PIN variable para legacy (por si acaso)
    esp_bt_sp_param_t param_type = ESP_BT_SP_IOCAP_MODE;
    esp_bt_io_cap_t iocap = ESP_BT_IO_CAP_NONE;
    esp_bt_gap_set_security_param(param_type, &iocap, sizeof(iocap));
    esp_bt_pin_code_t pin = {0};
    esp_bt_gap_set_pin(ESP_BT_PIN_TYPE_VARIABLE, 0, pin);

    esp_bt_gap_set_device_name("UnicornBridge");
    esp_bt_gap_set_scan_mode(ESP_BT_NON_CONNECTABLE, ESP_BT_NON_DISCOVERABLE);
    // Page timeout por defecto = 5.12 s: queda al límite si el Unicorn hace page scan en modo R2
    // (2.56 s) y no hay inquiry previo (sin clock offset). 0x4000 slots × 0.625 ms = 10.24 s.
    esp_bt_gap_set_page_timeout(0x4000);

    const uint8_t *own = esp_bt_dev_get_address();
    ESP_LOGI(TAG, "BT listo, MAC propia " ESP_BD_ADDR_STR, ESP_BD_ADDR_HEX(own));
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

#if CONFIG_BRIDGE_BT_DEBUG_LOGS
    // Eventos GAP/SPP propios en DEBUG; trazas internas de Bluedroid: menuconfig > Bluetooth > Log level
    esp_log_level_set(TAG, ESP_LOG_DEBUG);
#endif

    unsigned m[6];
    if (sscanf(CONFIG_UNICORN_MAC, "%x:%x:%x:%x:%x:%x", &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) != 6) {
        ESP_LOGE(TAG, "MAC inválida: %s", CONFIG_UNICORN_MAC);
        return;
    }
    for (int i = 0; i < 6; i++) {
        s_peer[i] = m[i];
    }
    ESP_LOGI(TAG, "Unicorn " ESP_BD_ADDR_STR " scn=%d (0=SDP) uart%d %d baud TX=GPIO%d",
             ESP_BD_ADDR_HEX(s_peer), CONFIG_UNICORN_SCN, BRIDGE_UART, CONFIG_BRIDGE_UART_BAUD,
             CONFIG_BRIDGE_UART_TX_GPIO);

    unicorn_parser_init(&s_parser);
    s_evq = xQueueCreate(16, sizeof(bridge_ev_t));
    s_sb = xStreamBufferCreate(SB_SIZE, 1);
    s_sync_q = xQueueCreate(8, sizeof(int64_t));
    t0_reset();

    init_bt();
    init_gpio_uart(); // después de init_bt: con espejo en consola, los logs se apagan aquí

    // Core 1 solo para el proceso (regla dura 2); control y LED con el BT en core 0
    xTaskCreatePinnedToCore(proc_task, "proc", 4096, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(led_task, "led", 2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(ctrl_task, "ctrl", 4096, NULL, 5, NULL, 0);
}
