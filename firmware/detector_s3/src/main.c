// INTUNE Detector (C2, ESP32-S3): EEG filtrado + pulsos de sincronía del brazo ->
// epochs ErrP -> ErrP-AE (TFLite Micro int8) -> nivel de alerta 0-3 -> brazo por UART.
//
// Consola (UART0/USB): logs ESP_LOG y una línea JSON por evento para el dashboard
// de C5 ({"ev":"epoch"|"stats"|"calib_start"|"calib_done"|"arm_event"|"confirm"|"selftest", ...}).
#include "arm_link.h"
#include "detector.h"
#include "errp_model.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

static const char *TAG = "intune";

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // El brazo recibe nivel 3 desde el primer momento (heartbeat activo antes que nada)
    arm_link_start();

    if (!errp_model_init()) {
        ESP_LOGE(TAG, "no se pudo iniciar el modelo: el brazo queda en paro seguro");
        vTaskDelay(portMAX_DELAY);
    }
#if CONFIG_DETECTOR_SELFTEST_ON_BOOT
    if (!detector_selftest()) {
        ESP_LOGW(TAG, "la prueba de epochs de referencia FALLÓ: revisar export del modelo");
    }
#endif
    detector_start();
    ESP_LOGI(TAG, "INTUNE detector en marcha");
}
