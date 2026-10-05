# ml/autoencoder (C2 Detector)

ErrP-AE (Keras → TFLite int8 con TFLite Micro en el S3) y baseline LDA, entrenados con el dataset
de C4 (`ml/data/FORMAT.md`, `intune-c4-dataset-1.0`). El firmware está en `firmware/detector_s3/`.

| Archivo | Qué hace |
|---|---|
| `errp_pipeline.py` | Lee las sesiones `.npz` de C4, normaliza con `norm_mean`/`norm_std` de cada sesión, split por sesión |
| `errp_ae.py` | Arquitecturas (conv EEGNet y denso de respaldo) y la cadena del S3 en numpy (bit a bit con `ae_*`) |
| `train_errp_ae.py` | Entrenamiento → `model.keras`, `config.json`, `rep.npz`, `val.npz` |
| `../c_exporter/export_to_c.py` | int8 + verificación (correlación ≥ 0.98) → `firmware/detector_s3/include/autoencoder_weights.h` |
| `baseline_lda.py` | LDA de 64 features → `models/lda.json` |
| `training/export_headers.py` | Gate, LDA e IIR del S3 → `firmware/detector_s3/include/errp_params.h`, `errp_golden.h` |
| `score.py` | Scores por acción (float, int8 bit a bit con el S3, LDA) en CSV para C4 |

## Flujo (en la PC de entrenamiento)

```bash
# sesiones de C4 en ml/data/processed/*.npz (o --data archivos/carpetas)
export TF_ENABLE_ONEDNN_OPTS=0
python ml/autoencoder/train_errp_ae.py --out runs/r1                 # [--arch dense] [--test-session S] [--val-session S]
python ml/c_exporter/export_to_c.py --model runs/r1/model.keras --config runs/r1/config.json \
    --rep runs/r1/rep.npz --val runs/r1/val.npz --tflite-out runs/r1/model.tflite
python ml/autoencoder/baseline_lda.py                                # [--test-session S]
python ml/autoencoder/training/export_headers.py
python ml/autoencoder/score.py ml/data/processed/*.npz --model runs/r1/model.keras \
    --tflite runs/r1/model.tflite --out scores.csv                   # para C4
cd firmware/detector_s3 && pio run -e engine_test -t upload -t monitor   # prueba 12 en la placa
pio run -t upload -t monitor                                             # app del detector
```

Con `uv`: `uv run --project ml/autoencoder python <script> ...`.

## Entregas para C4 (prueba 11)

| Qué | Dónde |
|---|---|
| LDA 64 features, shrinkage, `w[64]`, `b` | `baseline_lda.py` → `models/lda.json` |
| Modelo float e int8 | `runs/<r>/model.keras`, `runs/<r>/model.tflite` |
| Score por acción | `score.py` → `session, action_id, y, rejected, is_calibration, evt_counter, ae_float, ae_int8, lda` |
| Correlación float/int8 | la imprime y la exige `export_to_c.py` (`AE_INT8_SCORE_CORR` en `autoencoder_weights.h`) |

`ae_int8` usa la emulación entera de TFLite Micro de `export_to_c.py`: es el mismo número que da
`ae_infer` en el S3. Las rechazadas llevan NaN.

## Decisiones acordadas con C4 (FORMAT.md §9)

1. **Umbrales**: T1/T2/T3 = p90/p97/p99 del score de las épocas `is_calibration` limpias de la sesión
   evaluada, como el S3 en vivo. Las de calibración no entran al entrenamiento.
2. **Puerta en el S3 = FORMAT.md §3**, mismo orden y umbrales que `build_dataset.py`: flags (HELD, GAP,
   SETTLING, FILTER_RESET, UNFILTERED), flat (std de un canal de X < 0.05 µV), amplitude
   (max |X| > 100 µV sobre X con baseline y diezmado), gyro (rango pico a pico por eje en la ventana
   de 250, máximo de 3 ejes, > GATE_GYRO), counter continuo y overlap < 1.0 s. `saturated` no existe en
   el S3 (no hay r_*): una saturación real sale como amplitude.
3. **Calibración en el S3**: primeras 80 épocas limpias (`DETECTOR_CALIB_EPOCHS`), mean/std por canal
   (Welford, ddof 0) y percentiles con el motor `ae_*`; se guarda en NVS con el `model_id`.
4. **Jitter**: ±1 muestra a 50 Hz replicando el borde.
5. **LDA**: los 8 bins de 150–700 ms caen en la rejilla de 20 ms como `j = 8, 11, 14, 18, 21, 25, 28, 32, 35`
   (160–700 ms) sobre X en µV.
6. **EVENT_LATENCY_OFFSET**: 0 hasta la prueba 10 (`DETECTOR_EVENT_LATENCY_OFFSET_US`); se avisa a C4.

## Resultado con 4 sesiones sintéticas de C4 (solo plomería, no evidencia)

Test `synth_s4` con umbrales de su calibración: AE AUC ≈ 0.51–0.53, LDA AUC ≈ 0.81. Con datos reales
se decide si el S3 usa el AE o el LDA (`DETECTOR_USE_LDA`).
