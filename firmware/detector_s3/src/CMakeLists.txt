# Componente principal del detector C2. espressif/esp-tflite-micro llega por idf_component.yml.
#
# Dos apps en el mismo proyecto (platformio.ini):
#   detector    (default) app completa: entrada del puente, sync, gate, calibración, alertas, brazo.
#               Inferencia con errp_model.cc + errp_model_data.c (training/export_headers.py).
#   engine_test (-DDETECTOR_ENGINE_TEST=1) verificación en placa del motor ae_*
#               (autoencoder_engine.c + autoencoder_tflm.cc + ml/c_exporter/export_to_c.py):
#               test/test_c_engine.c define app_main con los vectores dorados y el benchmark.
if(DETECTOR_ENGINE_TEST)
    idf_component_register(SRCS "autoencoder_engine.c" "autoencoder_tflm.cc" "../test/test_c_engine.c"
                           INCLUDE_DIRS "../include" "../../common"
                           PRIV_REQUIRES esp_timer)
else()
    idf_component_register(SRCS "main.c" "detector.c" "eeg_input.c" "arm_link.c" "errp_dsp.c"
                                "errp_model.cc" "errp_model_data.c"
                           INCLUDE_DIRS "." "../include" "../../common"
                           REQUIRES driver esp_timer nvs_flash)
endif()

# Sin contracción FMA en la cadena float: score bit a bit con la referencia de Python (errp_ae)
if(DETECTOR_ENGINE_TEST)
    set_source_files_properties(autoencoder_engine.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
