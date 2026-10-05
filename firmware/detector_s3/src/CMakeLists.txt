idf_component_register(SRCS "main.c" "detector.c" "eeg_input.c" "arm_link.c" "errp_dsp.c"
                            "errp_model.cc" "errp_model_data.c"
                       INCLUDE_DIRS "." "../include" "../../common"
                       REQUIRES driver esp_timer nvs_flash)
