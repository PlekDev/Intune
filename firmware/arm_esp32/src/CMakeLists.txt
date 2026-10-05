idf_component_register(SRCS "main.c" "motion.c" "safety.c" "actions.c" "arm_uart.c"
                       INCLUDE_DIRS "." "../../common"
                       REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_driver_uart esp_driver_gpio esp_timer)
