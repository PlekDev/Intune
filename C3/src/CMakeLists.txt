idf_component_register(SRCS "main.c" "motion.c" "safety.c" "actions.c" "arm_uart.c" "detector.c"
                            # motor del ErrP-AE de C2 (TFLite Micro int8), compartido con el S3
                            "../../firmware/detector_s3/src/autoencoder_engine.c"
                            "../../firmware/detector_s3/src/autoencoder_tflm.cc"
                       INCLUDE_DIRS "." "../../firmware/common" "../../firmware/detector_s3/include"
                       REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_driver_uart esp_driver_gpio esp_timer)

# Como en el S3: sin contracción FMA en la cadena float (score igual que la referencia de Python)
set_source_files_properties("../../firmware/detector_s3/src/autoencoder_engine.c" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
