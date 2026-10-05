# Componente principal del detector C2.
# Por ahora la app es la verificación del motor en la placa: test/test_c_engine.c define app_main
# con ESP_PLATFORM (vectores dorados con TFLite Micro + ESP-NN y benchmark de la prueba 12).
# Cuando exista la app del detector, cambiar "../test/test_c_engine.c" por su main.c.
# espressif/esp-tflite-micro llega por idf_component.yml (gestor de componentes de ESP-IDF).
idf_component_register(SRCS "autoencoder_engine.c" "autoencoder_tflm.cc" "../test/test_c_engine.c"
                       INCLUDE_DIRS "../include" "../../common"
                       PRIV_REQUIRES esp_timer)

# Sin contracción FMA en la cadena float: score bit a bit con la referencia de Python (errp_ae)
set_source_files_properties(autoencoder_engine.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
