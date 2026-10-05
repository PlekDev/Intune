#include "arm_link.h"

#include <math.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "sdkconfig.h"

static const char *TAG = "arm_link";

#define ARM_UART UART_NUM_2

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static arm_alert_t s_alert = {.level = ARM_LEVEL_SEVERE, .action_id = 0xFFFF};  // arranca en paro seguro
static QueueHandle_t s_events;
static volatile bool s_confirm;
static arm_link_stats_t s_stats;
static arm_decoder_t s_dec;

static void send_alert(void)
{
    uint8_t out[sizeof(arm_alert_t) + 5];
    portENTER_CRITICAL(&s_mux);
    s_alert.seq++;
    arm_alert_t a = s_alert;
    portEXIT_CRITICAL(&s_mux);
    size_t n = arm_encode(ARM_MSG_ALERT, &a, sizeof(a), out);
    uart_write_bytes(ARM_UART, out, n);  // usa el buffer TX del driver: no bloquea
    s_stats.tx_frames++;
}

void arm_link_set_alert(uint8_t level, uint8_t flags, uint16_t action_id, float score)
{
    float q = score * 1000.0f;
    uint16_t score_q = isfinite(q) ? (uint16_t)fminf(fmaxf(q, 0.0f), 65535.0f) : 65535;
    portENTER_CRITICAL(&s_mux);
    bool changed = s_alert.level != level || s_alert.flags != flags || s_alert.action_id != action_id;
    s_alert.level = level;
    s_alert.flags = flags;
    s_alert.action_id = action_id;
    s_alert.score_q = score_q;
    portEXIT_CRITICAL(&s_mux);
    if (changed) {
        send_alert();
    }
}

static void heartbeat_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        send_alert();
        vTaskDelayUntil(&last, pdMS_TO_TICKS(ARM_HEARTBEAT_MS));
    }
}

static void rx_task(void *arg)
{
    (void)arg;
    uint8_t buf[64];
    for (;;) {
        int n = uart_read_bytes(ARM_UART, buf, sizeof(buf), pdMS_TO_TICKS(50));
        for (int i = 0; i < n; i++) {
            if (!arm_decode_byte(&s_dec, buf[i])) {
                continue;
            }
            uint8_t type = s_dec.buf[1], len = s_dec.buf[2];
            const uint8_t *p = s_dec.buf + 3;
            if (type == ARM_MSG_EVENT && len == sizeof(arm_event_t)) {
                arm_event_t ev;
                memcpy(&ev, p, sizeof(ev));
                xQueueSend(s_events, &ev, 0);  // si se llena, se pierde solo metadata de log
                s_stats.events++;
            } else if (type == ARM_MSG_CONFIRM) {
                s_confirm = true;
                s_stats.confirms++;
            }
        }
        s_stats.rx_frames = s_dec.frames;
        s_stats.rx_crc_errors = s_dec.crc_errors;
    }
}

bool arm_link_pop_event(arm_event_t *ev)
{
    return xQueueReceive(s_events, ev, 0) == pdTRUE;
}

bool arm_link_take_confirm(void)
{
    bool c = s_confirm;
    s_confirm = false;
    return c;
}

void arm_link_get_stats(arm_link_stats_t *out)
{
    *out = s_stats;
}

void arm_link_start(void)
{
    s_events = xQueueCreate(8, sizeof(arm_event_t));
    uart_config_t uc = {
        .baud_rate = ARM_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(ARM_UART, 1024, 1024, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(ARM_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(ARM_UART, CONFIG_DETECTOR_ARM_TX_GPIO, CONFIG_DETECTOR_ARM_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    xTaskCreatePinnedToCore(heartbeat_task, "arm_hb", 3072, NULL, 15, NULL, 0);
    xTaskCreatePinnedToCore(rx_task, "arm_rx", 3072, NULL, 11, NULL, 0);
    ESP_LOGI(TAG, "UART%d TX GPIO%d RX GPIO%d @ %d, heartbeat %d ms (arranca en nivel 3)",
             ARM_UART, CONFIG_DETECTOR_ARM_TX_GPIO, CONFIG_DETECTOR_ARM_RX_GPIO, ARM_UART_BAUD, ARM_HEARTBEAT_MS);
}
