# ml/autoencoder (C2 Detector)

ErrP-AE (Keras → TFLite int8 para el S3) y baseline LDA, entrenados con el dataset de C4
(`ml/data/FORMAT.md`, `intune-c4-dataset-1.0`). El firmware que los usa está en
`firmware/detector_s3/`.

## Correr

```bash
# sesiones de C4 en ml/data/processed/*.npz (o pasar archivos/carpetas como argumento)
uv run --project ml/autoencoder ml/autoencoder/baseline_lda.py
uv run --project ml/autoencoder ml/autoencoder/training/train.py          # --arch auto|conv|dense
uv run --project ml/autoencoder ml/autoencoder/training/export_tflite.py  # int8 + correlación >= 0.98
uv run --project ml/autoencoder ml/autoencoder/training/evaluate.py       # config.json
uv run --project ml/autoencoder ml/autoencoder/training/export_headers.py # headers del S3
uv run --project ml/autoencoder ml/autoencoder/score.py ml/data/processed/<sesion>.npz --out scores.csv
```

`--test-session` / `--val-session` eligen las sesiones; por defecto test = última, val = penúltima.
Con menos de 3 sesiones el split es cronológico dentro de la sesión y se avisa (solo para probar código).

## Entregas para C4 (prueba 11)

| Qué | Dónde |
|---|---|
| LDA 64 features, shrinkage, `w[64]`, `b` | `baseline_lda.py` → `models/lda.json` |
| Modelo float e int8 | `models/errp_ae.keras`, `models/errp_ae_int8.tflite` |
| Score por acción (float, int8, LDA) | `score.py` → CSV con `session, action_id, y, rejected, is_calibration, evt_counter, ae_float, ae_int8, lda` |
| Correlación float/int8 | `models/errp_ae_meta.json` → `tflite.float_int8_corr` (y `evaluation.float_int8_corr_test` en `config.json`) |

Score = MSE entre la época normalizada (norm_mean/norm_std de su sesión) y su reconstrucción, en float
tras decuantizar. Las rechazadas llevan NaN: el S3 no las puntúa.

## Decisiones acordadas con C4 (FORMAT.md §9)

1. **Umbrales**: T1/T2/T3 = p90/p97/p99 del score de las épocas `is_calibration` limpias de la sesión
   evaluada, como el S3 en vivo. Las de calibración no entran al entrenamiento.
2. **Puerta en el S3 = FORMAT.md §3**, mismo orden y umbrales que `build_dataset.py`: flags (HELD, GAP,
   SETTLING, FILTER_RESET, UNFILTERED), flat (std de un canal de X < 0.05 µV), amplitude
   (max |X| > 100 µV sobre X ya con baseline y diezmado), gyro (rango pico a pico por eje en la ventana
   de 250, máximo de 3 ejes, > GATE_GYRO), counter continuo y overlap < 1.0 s. `saturated` no existe en
   el S3 (no hay r_*): una saturación real sale como amplitude. Verificado en
   `firmware/detector_s3/test/host_dsp_test.c`.
3. **Calibración**: el S3 toma las primeras 80 épocas limpias (`DETECTOR_CALIB_EPOCHS`), std con piso
   1e-3 µV, igual que `build_dataset.py`.
4. **Jitter**: ±1 muestra a 50 Hz replicando el borde (no hace falta la ventana de 250).
5. **LDA**: los 8 bins de 150–700 ms caen en la rejilla de 20 ms como `j = 8, 11, 14, 18, 21, 25, 28, 32, 35`
   (160–700 ms) sobre X en µV.
6. **EVENT_LATENCY_OFFSET**: 0 hasta la prueba 10 (`DETECTOR_EVENT_LATENCY_OFFSET_US`); se avisa a C4.

## Resultado con 4 sesiones sintéticas de C4 (solo plomería, no evidencia)

Test `synth_s4` con umbrales de su calibración: AE AUC ≈ 0.51, LDA AUC ≈ 0.81. El AE no supervisado no
separa el ErrP sintético (su error de reconstrucción lo domina el fondo); el LDA sí. Correlación
float/int8 = 1.00. Con datos reales se decide si el S3 usa el AE o el LDA (`DETECTOR_USE_LDA`).
