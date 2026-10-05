#include "arm_uart.h"
#include "sdkconfig.h"
#if CONFIG_ARM_LINK_UART   // con ESP-NOW este archivo no aporta nada

#include <stdlib.h>
#include <string.h>
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "arm_uart";

#define ARM_UART   UART_NUM_1
#define FIFO_N     8
#define ARM_LINE_MAX   320

typedef struct {
    uint32_t hash;
    int64_t t_us;
    bool counts, verbose;
} pending_t;

static SemaphoreHandle_t s_lock;
static pending_t s_fifo[FIFO_N];
static int s_head, s_count;
static arm_uart_status_t s_st;
static int64_t s_fb_us;

static uint32_t hash_str(const char *p, size_t n)
{
    uint32_t h = 5381;
    while (n--) h = h * 33u ^ (uint8_t)*p++;
    return h;
}

static void pop(bool ok, int64_t now)
{
    pending_t p = s_fifo[s_head];
    s_head = (s_head + 1) % FIFO_N;
    s_count--;
    float ms = (now - p.t_us) / 1000.0f;
    if (ok) {
        s_st.echoes++;
        s_st.echo_ms_last = ms;
        if (ms > s_st.echo_ms_max) s_st.echo_ms_max = ms;
    } else {
        s_st.timeouts++;
    }
    if (p.counts) motion_on_ack(ok);
    if (p.verbose || (!ok && p.counts)) ESP_LOGI(TAG, "eco %s en %.1f ms", ok ? "OK" : "NO RECIBIDO", ms);
}

bool arm_uart_send(const char *json, bool counts_for_motion, bool verbose)
{
    size_t n = strlen(json);
    xSemaphoreTake(s_lock, portMAX_DELAY);
    if (s_count == FIFO_N) pop(false, esp_timer_get_time());    // el más viejo nunca tuvo eco
    int i = (s_head + s_count) % FIFO_N;
    s_fifo[i] = (pending_t){.hash = hash_str(json, n), .t_us = esp_timer_get_time(),
                            .counts = counts_for_motion, .verbose = verbose};
    s_count++;
    int w = uart_write_bytes(ARM_UART, json, n);
    w += uart_write_bytes(ARM_UART, "\n", 1);   // el brazo procesa al ver '\n'
    xSemaphoreGive(s_lock);
    if (verbose) ESP_LOGI(TAG, "-> %s", json);
    return w == (int)n + 1;
}

static bool num_after(const char *line, const char *key, float *out)
{
    const char *p = strstr(line, key);
    if (!p) return false;
    *out = strtof(p + strlen(key), NULL);
    return true;
}

static void handle_line(char *line, size_t n, int64_t now)
{
    if (!strncmp(line, "{\"T\":1051", 9)) {
        pose_t p;
        if (num_after(line, "\"b\":", &p.b) && num_after(line, "\"s\":", &p.s) &&
            num_after(line, "\"e\":", &p.e) && num_after(line, "\"t\":", &p.h)) {
            xSemaphoreTake(s_lock, portMAX_DELAY);
            s_st.pose = p;
            s_st.valid = true;
            s_st.fb_lines++;
            s_fb_us = now;
            xSemaphoreGive(s_lock);
        }
        return;
    }
    uint32_t h = hash_str(line, n);
    xSemaphoreTake(s_lock, portMAX_DELAY);
    int found = -1;
    for (int k = 0; k < s_count; k++)
        if (s_fifo[(s_head + k) % FIFO_N].hash == h) { found = k; break; }
    if (found >= 0) {
        for (int k = 0; k < found; k++) pop(false, now);   // anteriores sin eco: se perdieron
        pop(true, now);
    } else {
        s_st.other_lines++;
    }
    xSemaphoreGive(s_lock);
    if (found < 0) ESP_LOGI(TAG, "brazo: %s", line);   // arranque del brazo, avisos, etc.
}

static void rx_task(void *arg)
{
    static char line[ARM_LINE_MAX];
    size_t n = 0;
    bool overflow = false;
    uint8_t buf[128];
    for (;;) {
        int r = uart_read_bytes(ARM_UART, buf, sizeof buf, pdMS_TO_TICKS(10));
        int64_t now = esp_timer_get_time();
        if (r > 0) s_st.rx_bytes += (uint32_t)r;
        for (int i = 0; i < r; i++) {
            char c = (char)buf[i];
            if (c == '\n') {
                while (n && (line[n - 1] == '\r' || line[n - 1] == ' ')) n--;
                line[n] = 0;
                if (n && !overflow) handle_line(line, n, now);
                n = 0;
                overflow = false;
            } else if (n < ARM_LINE_MAX - 1) {
                line[n++] = c;
            } else {
                overflow = true;
            }
        }
        xSemaphoreTake(s_lock, portMAX_DELAY);
        while (s_count && now - s_fifo[s_head].t_us > ARM_UART_ECHO_MS * 1000LL) pop(false, now);
        xSemaphoreGive(s_lock);
    }
}

static void fb_task(void *arg)
{
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(ARM_UART_FB_MS));
        arm_uart_send("{\"T\":105}", false, false);
    }
}

void arm_uart_probe(int pin)
{
    gpio_reset_pin(pin);
    gpio_set_direction(pin, GPIO_MODE_INPUT);
    gpio_pulldown_en(pin);
    int high = 0, edges = 0, prev = gpio_get_level(pin);
    for (int i = 0; i < 1000; i++) {
        int v = gpio_get_level(pin);
        high += v;
        edges += v != prev;
        prev = v;
        vTaskDelay(1);   // 1 ms (FREERTOS_HZ = 1000)
    }
    gpio_pulldown_dis(pin);
    uart_set_pin(ARM_UART, CONFIG_ARM_LINK_TX_GPIO, pin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    ESP_LOGW(TAG, "probe GPIO%d: alto %d/1000 ms, cambios %d -> %s", pin, high, edges,
             high > 950 ? "línea en ALTO: compatible con un TX en reposo (o un pin de 3.3 V)"
             : high < 50 ? "línea en BAJO: pin suelto, GND o no es un TX"
             : "línea con actividad o a medio nivel");
}

arm_uart_status_t arm_uart_status(void)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    arm_uart_status_t r = s_st;
    r.age_ms = s_st.valid ? (esp_timer_get_time() - s_fb_us) / 1000 : -1;
    xSemaphoreGive(s_lock);
    return r;
}

void arm_uart_init(void)
{
    s_lock = xSemaphoreCreateMutex();
    uart_config_t uc = {
        .baud_rate = ARM_UART_BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(ARM_UART, 2048, 2048, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(ARM_UART, &uc));
    ESP_ERROR_CHECK(uart_set_pin(ARM_UART, CONFIG_ARM_LINK_TX_GPIO, CONFIG_ARM_LINK_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    xTaskCreate(rx_task, "arm_rx", 4096, NULL, 12, NULL);
    xTaskCreate(fb_task, "arm_fb", 3072, NULL, 9, NULL);
    ESP_LOGI(TAG, "enlace UART con el brazo: %d baud, TX=GPIO%d -> pin 8 del conector, RX=GPIO%d <- pin 10, GND <-> pin 6",
             ARM_UART_BAUD, CONFIG_ARM_LINK_TX_GPIO, CONFIG_ARM_LINK_RX_GPIO);
}

#endif  // CONFIG_ARM_LINK_UART
