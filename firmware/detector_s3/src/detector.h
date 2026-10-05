// Detector de ErrP: pulsos de sincronía -> epochs -> score -> nivel de alerta.
#pragma once
#include <stdbool.h>

// Prueba 12 parcial: los epochs de referencia dan el score int8 esperado.
bool detector_selftest(void);

void detector_start(void);
