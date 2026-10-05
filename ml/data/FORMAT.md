# Contrato del dataset ErrP (C4 → C2 y ml/eval/)

Versión del pipeline: `intune-c4-dataset-1.0`. Fuente de verdad de la cadena: CLAUDE.md, "Signal chain".
Si algo de aquí contradice al firmware, el firmware y CLAUDE.md mandan y se arregla este archivo.

## 1. Flujo

```
tools/recording/record_errp.py ──> ml/data/raw/<YYYYmmdd_HHMMSS>_<sujeto>/   (CSV crudos, .gitignore)
ml/data/make_synthetic.py      ──> ml/data/raw/<nombre>/                      (misma forma, datos falsos)
ml/data/build_dataset.py       ──> ml/data/processed/<sesion>.npz             (lo que lee C2; .gitignore)
```

Una sesión cruda = una carpeta con `eeg.csv`, `events.csv`, `blocks.csv`, `status.csv`, `meta.json`
(columnas y flags tal como las describe el prompt de C4; los bits de flags son los `LINK_F_*` de
`firmware/common/link_protocol.h`). `f_*` ya viene filtrado por el puente con el IIR causal 1–15 Hz:
**no se vuelve a filtrar** en ninguna etapa offline.

## 2. Cadena aplicada por época (idéntica a la del S3)

Para cada fila de `events.csv` (una acción):

1. **t = 0**: muestra Unicorn `c0 = evt_counter − round(EVENT_LATENCY_OFFSET / 4 ms)`. Se busca la fila de
   `eeg.csv` con `counter == c0`. El offset es parámetro (`--latency-offset-ms`, 0 hasta la prueba 10) y se guarda.
2. **Ventana**: muestras `−50 … +199` respecto a t = 0 (250 muestras = [−200, +800) ms) sobre `f_*`, en µV.
   Debe haber contador continuo (`+1` por fila) en toda la ventana.
3. **Baseline**: restar por canal la media de las muestras `−50 … −1` ([−200, 0) ms).
4. **Diezmado ×5**: sobre las muestras `0 … 199`, media de bloques de 5 (boxcar) → 40 muestras a 50 Hz.
   Muestra `j` (0…39) cubre `[20·j, 20·j + 20)` ms desde t = 0.
5. **Resultado `X[k]`** de forma `[8, 40]`, µV, **sin normalizar**. Canales en orden
   `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8` (orden del Unicorn y de `link_protocol.h`).

## 3. Puerta de artefactos (`rejected`, `reject_reason`)

`reject_reason[k]` es `"ok"` o los motivos que se cumplen unidos con `+`, en este orden de comprobación
(ej. `"flags+amplitude"`). `rejected = reject_reason != "ok"`. Una época rechazada **sí se guarda** en
`X`, salvo `status` / `counter` cuando no hay ventana (entonces `X[k]` es todo ceros).

| Motivo | Condición | Parámetro |
|---|---|---|
| `status` | `status != ok` en events.csv (`no_pulse`, `no_t0`, `overlap`, `arm_err`); sin `evt_counter` no hay época (X = 0) | — |
| `counter` | `c0` no existe en `eeg.csv` o el contador no es continuo en la ventana (X = 0 si no hay ventana) | — |
| `flags` | alguna muestra de la ventana con HELD, GAP, SETTLING o FILTER_RESET | — |
| `unfiltered` | alguna muestra con flag UNFILTERED (cadena distinta de la del S3) | — |
| `saturated` | algún `|r_*| ≥ 740 000 µV` en la ventana (tope del ADC ≈ 750 000). Solo offline; el S3 lo ve como amplitud enorme | `adc_sat_uv` |
| `flat` | algún canal de `X[k]` con desviación estándar < 0.05 µV | `flat_std_uv` |
| `amplitude` | `max |X[k]| > GATE_UV` (tras IIR, baseline y diezmado; 100 µV) | `gate_uv` |
| `gyro` | rango pico a pico por eje de `gyr_*` en la ventana, máximo de los 3 ejes, `> GATE_GYRO` °/s. No depende del sesgo del giroscopio y es fácil de calcular en el S3 | `gate_gyro` |
| `overlap` | otro flanco a < 1.0 s (250 muestras) antes o después, o `evt_flags & OVERLAP`. Se marcan las dos épocas del par | — |

`GATE_GYRO` empieza en 30 °/s. `build_dataset.py <sesion> --suggest-gyro` propone un valor con los bloques
`eyes_open` (p99 en reposo) y `head` (mediana), como media geométrica; con la sesión sintética da ≈ 13 °/s
(reposo 2.8, cabeza 63). Con datos reales, fijarlo con ese comando y anotarlo aquí.

## 4. Contenido del `.npz`

Se carga con `np.load(path, allow_pickle=False)`. Las cadenas son arreglos unicode; los escalares, 0-d
(`str(d["filter"])`, `int(d["fs_in"])`). `n` = número de acciones de la sesión (incluye rechazadas y manuales).

| Clave | Tipo / forma | Significado |
|---|---|---|
| `X` | float32 `[n, 8, 40]` | µV sin normalizar (§2) |
| `y` | int8 `[n]` | 0 correct, 1 error, −1 manual (label de events.csv) |
| `rejected` | bool `[n]` | puerta de §3 |
| `reject_reason` | str `[n]` | `"ok"` o motivos con `+` |
| `action_id` | int64 `[n]` | de events.csv |
| `evt_counter` | int64 `[n]` | contador del flanco (antes de restar el offset); −1 si no hubo pulso |
| `status` | str `[n]` | `ok`, `no_pulse`, `no_t0`, `overlap`, `arm_err` |
| `is_calibration` | bool `[n]` | épocas usadas para `norm_*` (§5) |
| `max_abs_uv` | float32 `[n]` | `max |X|`, para afinar `GATE_UV` (NaN si no hay ventana) |
| `gyro_metric` | float32 `[n]` | métrica de gyro de §3 (NaN si no hay ventana) |
| `norm_mean`, `norm_std` | float32 `[8]` | media/desv. por canal de las épocas de calibración (µV) |
| `norm_valid` | bool | `False` si hubo < 20 épocas de calibración (norm_* no fiables) |
| `n_calibration` | int64 | número de épocas de calibración |
| `ch_names` | str `[8]` | `Fz C3 Cz C4 Pz PO7 Oz PO8` |
| `subject`, `session` | str | sujeto y nombre de carpeta de la sesión |
| `fs_in`, `fs_out`, `decimation` | int64 | 250, 50, 5 |
| `window_ms`, `baseline_ms` | int64 `[2]` | `[-200, 800]`, `[-200, 0]` |
| `latency_offset_ms` | float64 | EVENT_LATENCY_OFFSET usado |
| `gate_uv`, `gate_gyro`, `flat_std_uv`, `adc_sat_uv` | float64 | parámetros de la puerta usados |
| `filter` | str | diseño del filtro tal como lo declara `meta.json` (o el `butter(2, [1, 15], ...)` canónico si falta) |
| `pipeline_version` | str | `intune-c4-dataset-1.0` |

### Vista aplanada (modelo denso)

`X_flat = X.reshape(n, 320)`, orden **canal-mayor**: índice `c * 40 + j` = canal `c`, muestra `j`
(canal 0 = Fz t0..t39, canal 1 = C3 t0..t39, …). Es el orden de `[8, 40]` en C (fila mayor), y lo que
debe recibir el AE denso 320→64→16→64→320 del S3. Se invierte con `.reshape(n, 8, 40)`.

## 5. Normalización y calibración

- `X` **no** viene normalizado. El z-score es **por canal** y **no por época** (borraría la amplitud):
  `Z = (X − norm_mean[None, :, None]) / norm_std[None, :, None]`.
- `norm_mean`/`norm_std` (8 valores cada uno) se calculan sobre las épocas de calibración de **esa sesión**:
  las primeras 80 acciones `y == 0 and not rejected` en orden de grabación (`--n-cal`), agrupando
  épocas y las 40 muestras de cada canal. `std` con piso de 1e-3 µV. Reproduce lo que hará el S3 con su bloque
  de calibración (se recalibra en cada sesión porque el contacto de los electrodos cambia).
- Para entrenar con varias sesiones, cada sesión se normaliza con **sus propias** estadísticas
  (así lo verá el S3 en vivo).

## 6. Reglas de uso (obligatorias)

1. El autoencoder se entrena **solo** con `y == 0 and not rejected`.
2. Las épocas de error (`y == 1`) son solo para evaluar y para el LDA. Las `manual` (`y == −1`) no se usan en entrenamiento ni en métricas.
3. El split train/val/test es **por sesión** (o por operador), nunca mezclando épocas de una misma sesión
   entre conjuntos. Con pocas sesiones, validación cruzada dejando una sesión fuera.
4. Las épocas con `is_calibration` no se usan para entrenar ni para elegir umbrales a mano en el
   desarrollo. En evaluación que simula el S3, los umbrales T1/T2/T3 de la sesión de prueba se calculan
   con **sus** épocas de calibración, como en vivo (ver "Decisiones abiertas").
5. Early stopping y selección de hiperparámetros con épocas correctas limpias de sesiones de validación, no de test.

## 7. Cómo correrlo

```bash
python ml/data/make_synthetic.py --name synth_s1            # sesión falsa → ml/data/raw/synth_s1/
python ml/data/build_dataset.py                             # todas las sesiones de raw/ → processed/*.npz
python ml/data/build_dataset.py ml/data/raw/<sesion> --latency-offset-ms 48 --gate-gyro 25
python ml/data/build_dataset.py ml/data/raw/<sesion> --suggest-gyro
python ml/eval/first_report.py ml/data/processed/synth_s1.npz   # rechazos + gran promedio + AUC
python -m pytest ml/data -q
```

Dependencias: numpy, scipy, pandas (y matplotlib, scikit-learn para el reporte). `make_synthetic.py` importa los
coeficientes de `tools/linux_probe/design_iir.py` (C1).

## 8. Para C2: de `X [n, 8, 40]` a la entrada del modelo

Entrada del ErrP-AE de CLAUDE.md: `[1, 8, 40]` (conv) o 320 valores canal-mayor (denso de respaldo).

```python
d = np.load("ml/data/processed/<sesion>.npz")
m = d["norm_mean"][None, :, None]; s = d["norm_std"][None, :, None]
keep = (d["y"] == 0) & ~d["rejected"] & ~d["is_calibration"]      # solo para entrenar
Z = ((d["X"] - m) / s)[keep].astype(np.float32)                    # [n, 8, 40]  → modelo conv: Z[:, None]
Zflat = Z.reshape(len(Z), 320)                                     # modelo denso
score = ((Z_hat - Z) ** 2).mean(axis=(1, 2))                       # MSE del epoch normalizado
```

Qué cambia respecto a la rama `origin/Autoencoder` (AE denso de 40 features, 40→16→6→16→40):

| | Rama Autoencoder (v1) | CLAUDE.md / este dataset |
|---|---|---|
| Entrada | 40 features | época `[8, 40]` = 320 valores (o `[1, 8, 40]` para conv) |
| Normalización | por feature | **por canal**: 8 medias y 8 desviaciones de la calibración de la sesión, no por época |
| Umbrales | p95 / p99 / p99.9 | **p90 / p97 / p99** (T1, T2, T3) del score MSE en épocas correctas de calibración; p99.5/p99.9 solo si C4 mide demasiadas paradas falsas por hora |
| Score | (el de la rama) | MSE entre época normalizada y reconstrucción, en float tras dequantizar |
| Arquitectura | denso 40→16→6→16→40 | conv EEGNet (~49 k parámetros) y denso 320→64→16→64→320 de respaldo |
| Datos | — | train solo `y == 0 and not rejected and not is_calibration`; split por sesión |

Aumentos de CLAUDE.md (jitter ±20 ms, ruido, ganancia ±10 % por canal): `X` ya está diezmado a 50 Hz
(1 muestra = 20 ms), así que el jitter de ±20 ms equivale a desplazar ±1 muestra de `X` (replicando el borde).
Si C2 quiere jitter más fino, que lo pida y C4 exporta también la ventana de 250 muestras.

## 9. Decisiones abiertas con C2

1. **Umbrales y calibración**: CLAUDE.md fija T1–T3 con las épocas correctas de calibración, no usadas en el
   entrenamiento. Este contrato las excluye del entrenamiento y las usa solo para `norm_*` y T1–T3 de la
   sesión evaluada. Confirmar que C2 evalúa así.
2. **Gate**: mismas definiciones que arriba en el S3 (amplitud sobre la época ya diezmada y con baseline; gyro como
   rango pico a pico por eje). Si el S3 la calcula distinto, avisar para cambiar `build_dataset.py` y no ambos.
3. **`GATE_GYRO`** (30 °/s de partida) y **`EVENT_LATENCY_OFFSET`** (0): se fijan con datos reales y la prueba 10.
4. **Errores en calibración**: el paradigma grabado mezcla errores desde el inicio; la calibración toma solo las
   primeras 80 correctas limpias. En vivo el S3 calibrará con acciones solo correctas.
5. **Falta `tools/recording/record_errp.py`**: C1 no lo subió a `origin/c1_bridge`. Este contrato sigue el formato crudo del
   prompt de C4; si el grabador real difiere (nombres de columnas, `meta.json`), se ajusta `build_dataset.py`.
