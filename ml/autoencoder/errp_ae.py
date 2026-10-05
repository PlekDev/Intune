#!/usr/bin/env python3
"""
ErrP-AE del detector C2 (INTUNE): modelos Keras y funciones de referencia en numpy.

Fuente única de las convenciones numéricas del motor C del ESP32-S3
(firmware/detector_s3/src/autoencoder_engine.c). La usan train_errp_ae.py, el
exportador ml/c_exporter/export_to_c.py y el pipeline offline (C4).

Importable SIN TensorFlow: las funciones de referencia solo usan numpy; Keras se
importa al construir un modelo (build_errp_conv_ae, build_dense_ae...).

Cadena de señal (CLAUDE.md del equipo, idéntica online y offline):
  ventana [-200, +800) ms a 250 Hz (250 muestras, la muestra j está en t = (j - 50) * 4 ms)
  -> línea base: media por canal de [-200, 0) ms (muestras 0..49)
  -> decimación x5: media de cada 5 muestras de [0, 800) ms (250 -> 50 Hz)   preprocess_window
  -> época 8 x 40 en uV, canales Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
  -> z-score por canal con la media/std de calibración del operador          normalize
  -> cuantización int8 con la escala/zero point del tensor de entrada        quantize
  -> ErrP-AE int8 (TFLite Micro) -> descuantización                          dequantize
  -> score = MSE entre z y su reconstrucción, en float                       score
  -> nivel 0..3 con los umbrales T1 <= T2 <= T3 (estricto ">")               level
Calibración por operador y sesión: media/std por canal sobre todas las muestras de
las épocas de calibración (norm_stats) y umbrales = percentiles 90/97/99 de los
scores de épocas correctas no usadas para entrenar (thresholds_from_scores).

Aritmética (bit a bit igual que el C, salvo norm_stats y thresholds_from_scores, que calculan en
float64 y redondean a float32: el C, en float32, puede diferir en ~1 ulp; comparar con tolerancia):
  float32 en todo; sumas en orden de índice (bucle secuencial, NO la suma por pares de
  np.sum) y sin multiplicación-suma fusionada; media = suma / cuenta;
  z      = (epoch - mean) / std
  q_in   = clamp(rint(z / s_in) + zp_in, -128, 127)       rint: mitad al par (= lrintf)
  z_hat  = s_out * (float)(q_out - zp_out)
  score  = (sum_i (z_i - z_hat_i)^2) / 320                orden [electrodo][tiempo]
En el S3, -ffp-contract=off (o equivalente) evita que el compilador fusione d * d + acc.

Modelos (entrada y salida NHWC (8, 40, 1) = [electrodo][tiempo], igual que float epoch[8][40]):
  errp_conv  EEGNet-style, ~49k parámetros, ~76k MACs:
             Conv2D 8 x (1, 9) same + BN + ReLU -> DepthwiseConv2D (8, 1) x2 + BN + ReLU
             -> AvgPool (1, 2) -> Flatten -> Dense 16 (latente, lineal) -> Dense 128 ReLU
             -> Dense 320 (lineal) -> Reshape (8, 40, 1)
  dense      alternativa: 320 -> 64 ReLU -> 16 (lineal) -> 64 ReLU -> 320 (lineal), ~43k parámetros
Ops TFLite tras la conversión int8 (BN se pliega en la conv y ReLU se fusiona): solo
CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RESHAPE y RELU, SIEMPRE que
se convierta con lote fijo 1 (inference_model): con el lote dinámico de Keras, Flatten y
Reshape arrastran SHAPE, STRIDED_SLICE y PACK.

Uso:
  import errp_ae as ae
  epoch = ae.preprocess_window(win)                    # [..., 8, 250] -> [..., 8, 40]
  mean, std = ae.norm_stats(calib_epochs)              # [N, 8, 40] -> [8], [8]
  z = ae.normalize(epoch, mean, std)
  t1, t2, t3 = ae.thresholds_from_scores(scores)       # p90, p97, p99
  model = ae.build_model("errp_conv")                  # requiere TensorFlow
  python ml/autoencoder/errp_ae.py --selftest          # comprueba las funciones de referencia (numpy)
  python ml/autoencoder/errp_ae.py --selftest --keras  # + construye los dos modelos (TensorFlow)
  python ml/autoencoder/errp_ae.py --summary errp_conv # model.summary(), parámetros y MACs
Determinismo con TensorFlow (entrenamiento, exportación): pin_single_cpu() ANTES de importarlo y
TF_ENABLE_ONEDNN_OPTS=0 (keras_module() lo fija por defecto).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Sequence, Tuple

import numpy as np

# ---- Constantes del contrato (CONTRACT_v3 §1, §4) ----

CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]  # orden Unicorn
FS_HZ = 250                 # Hz de la ventana cruda
DECIM = 5                   # 250 -> 50 Hz (media de cada 5 muestras)
PRE = 50                    # muestras de [-200, 0) ms (línea base)
POST = 200                  # muestras de [0, 800) ms
WIN = 250                   # PRE + POST
N_CH = 8
N_T = 40                    # POST / DECIM
N_IN = N_CH * N_T           # 320 = INPUT_DIM = OUTPUT_DIM
EPOCH_MS = (-200, 800)
BASELINE_MS = (-200, 0)
DEFAULT_PERCENTILES = (90.0, 97.0, 99.0)  # T1, T2, T3 (TUNE: p99.5 / p99.9)
CALIB_MIN_SCORES = 20       # mínimo de scores para calcular umbrales (AE_CALIB_MIN_SCORES)

INPUT_SHAPE = (N_CH, N_T, 1)  # NHWC sin el lote
ARCHS = ("errp_conv", "dense")
MODEL_NAMES = {"errp_conv": "errp_conv_ae", "dense": "dense_ae"}
EXPECTED_PARAMS = {"errp_conv": 48888, "dense": 43472}  # model.count_params() (incluye BN)
EXPECTED_MACS = {"errp_conv": 76288, "dense": 43008}    # multiplicaciones-acumulaciones conv/dense
# Ops TFLite permitidas en el modelo convertido (CONTRACT_v3 §2): el exportador rechaza el resto
ALLOWED_TFLITE_OPS = ("CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "FULLY_CONNECTED",
                      "RESHAPE", "RELU")

assert PRE + POST == WIN and POST == DECIM * N_T


# ---- Validación ----

def _check_scale(scale: float) -> np.float32:
    """Escala de cuantización float32 finita y > 0."""
    with np.errstate(over="ignore"):
        s = np.float32(scale)
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"escala de cuantización no válida: {scale!r} (debe ser float32 finito > 0)")
    return s


def _check_zero_point(zero_point: int) -> int:
    """Zero point int8 entero en [-128, 127]."""
    try:
        zp = int(zero_point)
        ok = zp == zero_point
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok or not -128 <= zp <= 127:
        raise ValueError(f"zero point no válido: {zero_point!r} (debe ser un entero en [-128, 127])")
    return zp


def check_percentiles(percentiles: Sequence[float]) -> Tuple[float, float, float]:
    try:
        p = tuple(float(v) for v in percentiles)
    except (TypeError, ValueError):
        raise ValueError(f"percentiles no válidos: {percentiles!r}") from None
    if len(p) != 3 or not all(math.isfinite(v) for v in p) or not 0.0 < p[0] < p[1] < p[2] <= 100.0:
        raise ValueError(f"percentiles no válidos: {percentiles!r} (3 valores con 0 < p1 < p2 < p3 <= 100)")
    return p  # type: ignore[return-value]


def _as_f32(x, what: str) -> np.ndarray:
    """Array float32 (lo que recibe el C); fuera de rango -> inf sin RuntimeWarning."""
    try:
        with np.errstate(over="ignore"):
            return np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: no es un array numérico ({e})") from None


def _rows320(a: np.ndarray, what: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """[..., 8, 40], [..., 8, 40, 1] o [..., 320] -> ([..., 320], forma del lote)."""
    if a.ndim >= 2 and a.shape[-2:] == (N_CH, N_T):
        batch = a.shape[:-2]
    elif a.ndim >= 3 and a.shape[-3:] == INPUT_SHAPE:
        batch = a.shape[:-3]
    elif a.ndim >= 1 and a.shape[-1] == N_IN:
        batch = a.shape[:-1]
    else:
        raise ValueError(f"{what}: forma {a.shape}; se esperaba [..., 8, 40], [..., 8, 40, 1] o [..., 320]")
    return a.reshape(batch + (N_IN,)), tuple(batch)


def percentile_key(p: float) -> str:
    """Clave de config.json["thresholds"] para un percentil: 90 -> "p90", 99.5 -> "p99.5"."""
    return "p" + format(float(p), "g")


# ---- Funciones de referencia (numpy, iguales al motor C) ----

def preprocess_window(win) -> np.ndarray:
    """Ventana cruda [..., 8, 250] (uV tras el IIR causal) -> época [..., 8, 40] float32.

    Igual que ae_preprocess: baseline[c] = media(win[c][0..49]);
    epoch[c][k] = media(win[c][50 + 5k .. 54 + 5k]) - baseline[c]. Sumas float32
    secuenciales en orden de índice, media = suma / cuenta. Los no finitos se propagan
    (el motor los detecta después: fail-safe)."""
    w = _as_f32(win, "ventana")
    if w.ndim < 2 or w.shape[-2:] != (N_CH, WIN):
        raise ValueError(f"ventana: forma {w.shape}; se esperaba [..., {N_CH}, {WIN}]")
    lead = w.shape[:-1]
    out = np.empty(lead + (N_T,), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        acc = np.zeros(lead, dtype=np.float32)
        for j in range(PRE):
            acc = acc + w[..., j]
        base = acc / np.float32(PRE)
        for k in range(N_T):
            acc = np.zeros(lead, dtype=np.float32)
            for j in range(PRE + DECIM * k, PRE + DECIM * (k + 1)):
                acc = acc + w[..., j]
            out[..., k] = acc / np.float32(DECIM) - base
    return out


def norm_stats(epochs) -> Tuple[np.ndarray, np.ndarray]:
    """Media y desviación típica POBLACIONAL (ddof 0) por canal sobre todas las muestras de
    todas las épocas [N, 8, 40] -> (mean[8], std[8]) float32.

    Calculado en float64 (dos pasadas) y redondeado a float32; el C usa Welford en
    float32 (diferencias ~1e-6 relativas). ValueError si hay no finitos o algún canal
    queda con std <= 0 (canal plano): esa calibración sería inválida."""
    e = _as_f32(epochs, "épocas")
    if e.ndim != 3 or e.shape[1:] != (N_CH, N_T) or e.shape[0] < 1:
        raise ValueError(f"épocas: forma {e.shape}; se esperaba [N >= 1, {N_CH}, {N_T}]")
    if not np.all(np.isfinite(e)):
        raise ValueError("épocas con valores no finitos: no se pueden calcular media/std")
    e64 = e.astype(np.float64)
    mean = e64.mean(axis=(0, 2))
    std = e64.std(axis=(0, 2))  # ddof = 0
    with np.errstate(over="ignore"):
        mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    bad = [CHANNELS[c] for c in range(N_CH)
           if not (np.isfinite(mean32[c]) and np.isfinite(std32[c]) and std32[c] > 0)]
    if bad:
        raise ValueError(f"media/std no válidas en los canales {bad} (std <= 0: canal plano)")
    return mean32, std32


def normalize(epochs, mean, std) -> np.ndarray:
    """z = (epoch - mean[c]) / std[c] en float32 (resta y división float32, como ae_normalize).

    epochs [..., 8, 40]; mean/std [8] o [..., 8] (estadísticas por época, p.ej. por sesión).
    ValueError si mean/std no son finitas o algún std <= 0. Los no finitos de las épocas se
    propagan a z (el motor C los convierte en fail-safe)."""
    e = _as_f32(epochs, "épocas")
    if e.ndim < 2 or e.shape[-2:] != (N_CH, N_T):
        raise ValueError(f"épocas: forma {e.shape}; se esperaba [..., {N_CH}, {N_T}]")
    m = _as_f32(mean, "mean")
    s = _as_f32(std, "std")
    if m.ndim < 1 or s.ndim < 1 or m.shape[-1] != N_CH or s.shape[-1] != N_CH:
        raise ValueError(f"mean/std: formas {m.shape} y {s.shape}; se esperaba [..., {N_CH}]")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError("calibración inválida: mean/std no finitas o std <= 0")
    with np.errstate(over="ignore", invalid="ignore"):
        z = (e - m[..., None]) / s[..., None]
    return z.astype(np.float32, copy=False)


def quantize(z, scale: float, zero_point: int) -> np.ndarray:
    """q = clamp(rint(z / scale) + zero_point, -128, 127) -> int8 (división float32, redondeo
    mitad al par como lrintf; satura para cualquier z finito). ValueError si z no es finito:
    el motor C comprueba z antes de cuantizar (fail-safe) y lrintf(inf/NaN) no está definido."""
    s = _check_scale(scale)
    zp = _check_zero_point(zero_point)
    a = _as_f32(z, "z")
    if not np.all(np.isfinite(a)):
        raise ValueError("z con valores no finitos: no se cuantiza (fail-safe del motor)")
    with np.errstate(over="ignore"):
        q = np.rint(a / s) + np.float32(zp)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(q, scale: float, zero_point: int) -> np.ndarray:
    """x = scale * (float)(q - zero_point) en float32 (resta entera, producto float32)."""
    s = _check_scale(scale)
    zp = _check_zero_point(zero_point)
    qa = np.asarray(q)
    if not np.issubdtype(qa.dtype, np.integer):
        raise ValueError(f"q debe ser entero (int8), no {qa.dtype}")
    if qa.size and (int(qa.min()) < -128 or int(qa.max()) > 127):
        raise ValueError("q fuera del rango int8 [-128, 127]")
    d = (qa.astype(np.int32) - np.int32(zp)).astype(np.float32)
    return (s * d).astype(np.float32, copy=False)


def score(z, z_hat):
    """MSE por época entre z y z_hat (320 valores en orden [electrodo][tiempo]) en float32:
    acc += d * d secuencial (sin FMA) y score = acc / 320. Acepta [..., 8, 40],
    [..., 8, 40, 1] (salida Keras) o [..., 320]; devuelve np.float32 o un array [...]. Los no
    finitos se propagan (ae_infer devuelve +INFINITY en ese caso)."""
    a, batch = _rows320(_as_f32(z, "z"), "z")
    b, batch_b = _rows320(_as_f32(z_hat, "z_hat"), "z_hat")
    if batch != batch_b:
        raise ValueError(f"z y z_hat con lotes distintos: {batch} y {batch_b}")
    acc = np.zeros(batch, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(N_IN):
            d = a[..., i] - b[..., i]
            acc = acc + d * d
        out = acc / np.float32(N_IN)
    return np.float32(out) if out.ndim == 0 else out.astype(np.float32, copy=False)


def level(score, t1: float, t2: float, t3: float):
    """Nivel de alerta por score (CLAUDE.md): 0 si score <= T1; 1 si T1 < score <= T2;
    2 si T2 < score <= T3; 3 si score > T3 (">" estricto, comparaciones en float32).
    Fail-safe -> 3: score no finito o umbrales inválidos (no finitos o no 0 < T1 <= T2 <= T3).
    Escalar -> int; array -> array int64 de la misma forma."""
    with np.errstate(over="ignore"):
        t = np.array([t1, t2, t3], dtype=np.float64).astype(np.float32)
    valid = bool(np.all(np.isfinite(t)) and 0 < t[0] <= t[1] <= t[2])
    s = _as_f32(score, "score")  # (el parámetro tapa la función score() aquí: no se usa)
    if not valid:
        out = np.full(s.shape, 3, dtype=np.int64)
    else:
        out = np.where(s > t[2], 3, np.where(s > t[1], 2, np.where(s > t[0], 1, 0))).astype(np.int64)
        out[~np.isfinite(s)] = 3
    return int(out) if out.ndim == 0 else out


def thresholds_from_scores(scores, percentiles: Sequence[float] = DEFAULT_PERCENTILES
                           ) -> Tuple[float, float, float]:
    """Umbrales (T1, T2, T3) = percentiles "linear" de numpy de los scores (float32):
    s ordenados, pos = p/100 * (n - 1), lo = floor(pos), hi = min(lo + 1, n - 1),
    T = s[lo] + (pos - lo) * (s[hi] - s[lo]) (en float64, redondeado a float32).

    Requiere n >= 20 scores, todos finitos, y que resulte 0 < T1 <= T2 <= T3 (ValueError si
    no). Devuelve floats de Python exactamente representables en float32."""
    p = check_percentiles(percentiles)
    s = _as_f32(scores, "scores").ravel()
    if s.size < CALIB_MIN_SCORES:
        raise ValueError(f"hacen falta >= {CALIB_MIN_SCORES} scores para los umbrales (hay {s.size})")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores con valores no finitos: no se calculan umbrales")
    s = np.sort(s)
    n = int(s.size)
    out = []
    for pct in p:
        pos = (pct / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        a, b = float(s[lo]), float(s[hi])
        out.append(float(np.float32(a + (pos - lo) * (b - a))))
    t1, t2, t3 = out
    if not 0.0 < t1 <= t2 <= t3:
        raise ValueError(f"umbrales no válidos: T1={t1!r} T2={t2!r} T3={t3!r} (se exige 0 < T1 <= T2 <= T3)")
    return t1, t2, t3


# ---- Modelos Keras (TensorFlow solo aquí) ----

def pin_single_cpu() -> str:
    """Fija el proceso a UNA CPU lógica (la de menor número de las permitidas). Llamarla ANTES de
    importar TensorFlow, en el punto de entrada (entrenamiento, exportación).

    Motivo: en CPUs híbridas (Intel 12.ª gen. o posterior, núcleos P y E), Eigen elige el
    blocking de sus GEMM según la caché del núcleo donde corre la primera multiplicación; dos
    procesos con los mismos datos dan floats distintos en los últimos bits (y pesos entrenados
    distintos) según caigan en un núcleo P o E. Fijado a una CPU: misma máquina + mismas entradas
    -> mismos bits. Devuelve "CPU n" o "" si no se pudo (p.ej. macOS)."""
    try:
        if hasattr(os, "sched_setaffinity"):  # Linux
            cpu = min(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpu})
            return f"CPU {cpu}"
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
            k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            proc = k32.GetCurrentProcess()
            allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not k32.GetProcessAffinityMask(proc, ctypes.byref(allowed), ctypes.byref(system)) or not allowed.value:
                return ""
            lowest = allowed.value & -allowed.value
            if not k32.SetProcessAffinityMask(proc, lowest):
                return ""
            return f"CPU {lowest.bit_length() - 1}"
    except (OSError, AttributeError, ValueError):
        return ""
    return ""


def keras_module():
    """Importa Keras 3 con backend TensorFlow. Antes de importar TensorFlow fija por defecto
    TF_ENABLE_ONEDNN_OPTS=0 (oneDNN cambia los resultados float entre ejecuciones: el
    entrenamiento, la exportación y los vectores dorados deben ser deterministas) y
    TF_CPP_MIN_LOG_LEVEL=2; un valor ya presente en el entorno se respeta."""
    if "tensorflow" not in sys.modules:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if "keras" not in sys.modules:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import keras
    except ImportError:
        raise ImportError("falta TensorFlow/Keras en este entorno de Python "
                          "(pip install tensorflow-cpu): hace falta para construir el modelo") from None
    backend = keras.backend.backend()
    if backend != "tensorflow":
        raise RuntimeError(f"Keras usa el backend {backend!r}; el ErrP-AE requiere TensorFlow "
                           "(KERAS_BACKEND=tensorflow) para convertirlo a TFLite")
    return keras


def _check_l2(l2: float) -> float:
    try:
        v = float(l2)
    except (TypeError, ValueError):
        raise ValueError(f"l2 no válido: {l2!r}") from None
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"l2 no válido: {l2!r} (debe ser finito y >= 0)")
    return v


def _finish(model, arch: str):
    """Comprueba formas y número de parámetros contra el contrato (evita cambios accidentales)."""
    expected = (None,) + INPUT_SHAPE
    if tuple(model.input_shape) != expected or tuple(model.output_shape) != expected:
        raise RuntimeError(f"{arch}: formas {model.input_shape} -> {model.output_shape}; "
                           f"se esperaba {expected} -> {expected}")
    n = model.count_params()
    if n != EXPECTED_PARAMS[arch]:
        raise RuntimeError(f"{arch}: {n} parámetros; el contrato fija {EXPECTED_PARAMS[arch]}")
    return model


def build_errp_conv_ae(l2: float = 1e-4):
    """ErrP-AE EEGNet-style del CLAUDE.md (48888 parámetros, 76288 MACs):
      enc1  Conv2D 8 filtros (1, 9) same, sin bias + BatchNorm + ReLU     -> (8, 40, 8)
      enc2  DepthwiseConv2D (8, 1) x2, sin bias + BatchNorm + ReLU        -> (1, 40, 16)
      enc3  AveragePooling2D (1, 2)                                       -> (1, 20, 16)
      enc4  Flatten + Dense 16 lineal (latente)                           -> 16
      dec1  Dense 128 ReLU                                                -> 128
      dec2  Dense 320 lineal + Reshape                                    -> (8, 40, 1)
    Sin bias en las convs (EEGNet): BatchNorm ya aporta el desplazamiento y al convertir se
    pliega en la conv. l2: penalización L2 (weight decay) de los kernels; 0 la desactiva."""
    keras = keras_module()
    L = keras.layers
    l2 = _check_l2(l2)

    def reg():
        return keras.regularizers.L2(l2) if l2 > 0 else None

    inp = keras.Input(shape=INPUT_SHAPE, name="z")
    x = L.Conv2D(8, (1, 9), padding="same", use_bias=False, kernel_regularizer=reg(), name="enc1_conv")(inp)
    x = L.BatchNormalization(name="enc1_bn")(x)
    x = L.ReLU(name="enc1_relu")(x)
    x = L.DepthwiseConv2D((N_CH, 1), depth_multiplier=2, padding="valid", use_bias=False,
                          depthwise_regularizer=reg(), name="enc2_dwconv")(x)
    x = L.BatchNormalization(name="enc2_bn")(x)
    x = L.ReLU(name="enc2_relu")(x)
    x = L.AveragePooling2D((1, 2), name="enc3_pool")(x)
    x = L.Flatten(name="enc4_flatten")(x)
    x = L.Dense(16, kernel_regularizer=reg(), name="enc4_latent")(x)
    x = L.Dense(128, activation="relu", kernel_regularizer=reg(), name="dec1_dense")(x)
    x = L.Dense(N_IN, kernel_regularizer=reg(), name="dec2_dense")(x)
    out = L.Reshape(INPUT_SHAPE, name="z_hat")(x)
    return _finish(keras.Model(inp, out, name=MODEL_NAMES["errp_conv"]), "errp_conv")


def build_dense_ae(l2: float = 1e-4):
    """Alternativa densa del CLAUDE.md (43472 parámetros, 43008 MACs), misma interfaz:
    Flatten 320 -> Dense 64 ReLU -> Dense 16 lineal (latente) -> Dense 64 ReLU
    -> Dense 320 lineal -> Reshape (8, 40, 1)."""
    keras = keras_module()
    L = keras.layers
    l2 = _check_l2(l2)

    def reg():
        return keras.regularizers.L2(l2) if l2 > 0 else None

    inp = keras.Input(shape=INPUT_SHAPE, name="z")
    x = L.Flatten(name="flatten")(inp)
    x = L.Dense(64, activation="relu", kernel_regularizer=reg(), name="enc1_dense")(x)
    x = L.Dense(16, kernel_regularizer=reg(), name="enc2_latent")(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=reg(), name="dec1_dense")(x)
    x = L.Dense(N_IN, kernel_regularizer=reg(), name="dec2_dense")(x)
    out = L.Reshape(INPUT_SHAPE, name="z_hat")(x)
    return _finish(keras.Model(inp, out, name=MODEL_NAMES["dense"]), "dense")


def build_model(arch: str, l2: float = 1e-4):
    """build_errp_conv_ae o build_dense_ae según arch ("errp_conv" | "dense")."""
    if arch == "errp_conv":
        return build_errp_conv_ae(l2)
    if arch == "dense":
        return build_dense_ae(l2)
    raise ValueError(f"arquitectura desconocida: {arch!r} (opciones: {', '.join(ARCHS)})")


def inference_model(model):
    """Modelo equivalente con lote FIJO 1 (entrada [1, 8, 40, 1], modo inferencia: BatchNorm con
    sus medias móviles) para convertir a TFLite. Con el lote dinámico (None), Flatten y Reshape
    de Keras 3 calculan tf.shape(x)[0] y el .tflite arrastra SHAPE, STRIDED_SLICE y PACK (ops no
    permitidas); con lote 1 solo quedan CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D,
    FULLY_CONNECTED y RESHAPE. Comparte los pesos con model (no los copia).
    Uso: tf.lite.TFLiteConverter.from_keras_model(inference_model(model))."""
    keras = keras_module()
    x = keras.Input(batch_shape=(1,) + INPUT_SHAPE, name="z")
    return keras.Model(x, model(x, training=False), name=f"{model.name}_b1")


def count_macs(model) -> int:
    """Multiplicaciones-acumulaciones de Conv2D, DepthwiseConv2D y Dense por época (BatchNorm
    se pliega al convertir; AveragePooling2D y ReLU no cuentan). Recorre submodelos."""
    total = 0
    for layer in model.layers:
        kind = type(layer).__name__
        if hasattr(layer, "layers"):  # submodelo (p.ej. inference_model)
            total += count_macs(layer)
        elif kind == "Conv2D":
            _, h, w, f = layer.output.shape
            kh, kw = layer.kernel_size
            total += int(h) * int(w) * int(f) * int(kh) * int(kw) * int(layer.input.shape[-1])
        elif kind == "DepthwiseConv2D":
            _, h, w, c = layer.output.shape
            kh, kw = layer.kernel_size
            total += int(h) * int(w) * int(c) * int(kh) * int(kw)
        elif kind == "Dense":
            total += int(layer.input.shape[-1]) * int(layer.units)
    return total


# ---- Autotest (numpy; con --keras también los modelos) ----

class SelftestFailure(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SelftestFailure(msg)


def _naive_preprocess(win: np.ndarray) -> np.ndarray:
    """Bucles explícitos escalar a escalar en float32 (como el C)."""
    out = np.empty((N_CH, N_T), dtype=np.float32)
    for c in range(N_CH):
        acc = np.float32(0.0)
        for j in range(PRE):
            acc = np.float32(acc + win[c, j])
        base = np.float32(acc / np.float32(PRE))
        for k in range(N_T):
            acc = np.float32(0.0)
            for j in range(DECIM):
                acc = np.float32(acc + win[c, PRE + DECIM * k + j])
            out[c, k] = np.float32(np.float32(acc / np.float32(DECIM)) - base)
    return out


def _selftest_numpy() -> list:
    steps = []
    rng = np.random.default_rng(20261004)

    # Preprocesado: bit a bit contra bucles escalares (offset DC grande + ruido en uV)
    wins = (rng.normal(0.0, 15.0, (12, N_CH, WIN)) + rng.uniform(-300, 300, (12, N_CH, 1))).astype(np.float32)
    got = preprocess_window(wins)
    _check(got.shape == (12, N_CH, N_T) and got.dtype == np.float32, "preprocess_window: forma/tipo")
    for i in range(wins.shape[0]):
        _check(np.array_equal(got[i].view(np.uint32), _naive_preprocess(wins[i]).view(np.uint32)),
               f"preprocess_window != bucles explícitos (ventana {i})")
    _check(np.array_equal(preprocess_window(wins[3]), got[3]), "preprocess_window de una sola ventana")
    ramp = np.tile(np.arange(WIN, dtype=np.float32), (N_CH, 1))  # baseline 24.5; bloque k: 52 + 5k
    _check(np.array_equal(preprocess_window(ramp)[0], 27.5 + 5.0 * np.arange(N_T, dtype=np.float32)),
           "preprocess_window: rampa (índices de la línea base o de la decimación desplazados)")
    steps.append("preprocess_window: bit a bit igual a bucles escalares float32 (12 ventanas) y rampa exacta")

    # norm_stats: np.mean / np.std (ddof 0) y Welford en float64
    ep = (rng.normal(0.0, 1.0, (57, N_CH, N_T)) * rng.uniform(2, 30, (1, N_CH, 1))
          + rng.uniform(-5, 5, (1, N_CH, 1))).astype(np.float32)
    m, s = norm_stats(ep)
    flat = ep.transpose(1, 0, 2).reshape(N_CH, -1).astype(np.float64)
    _check(np.allclose(m, flat.mean(axis=1), rtol=1e-6, atol=1e-6)
           and np.allclose(s, np.std(flat, axis=1, ddof=0), rtol=1e-6, atol=0), "norm_stats != np.mean/np.std")
    for c in range(N_CH):
        n_w, mean_w, m2 = 0, 0.0, 0.0
        for v in flat[c]:
            n_w += 1
            delta = v - mean_w
            mean_w += delta / n_w
            m2 += delta * (v - mean_w)
        _check(abs(mean_w - m[c]) <= 1e-6 * (1 + abs(mean_w)) and abs(math.sqrt(m2 / n_w) - s[c]) <= 1e-6 * s[c],
               f"norm_stats != Welford (canal {c})")
    flat_ch = ep.copy()
    flat_ch[:, 5, :] = 7.0
    try:
        norm_stats(flat_ch)
        raise SelftestFailure("norm_stats aceptó un canal plano")
    except ValueError as e:
        _check("PO7" in str(e), f"norm_stats: mensaje inesperado: {e}")
    steps.append("norm_stats: igual a np.mean/np.std(ddof=0), Welford float64 (1e-6) y rechaza un canal plano")

    # normalize: float32 explícito, con estadísticas [8] y por época [N, 8]
    z = normalize(ep, m, s)
    _check(np.array_equal(z, (ep - m[None, :, None]) / s[None, :, None]), "normalize con mean/std [8]")
    mi = np.repeat(m[None], ep.shape[0], axis=0)
    si = np.repeat(s[None], ep.shape[0], axis=0)
    _check(np.array_equal(normalize(ep, mi, si), z), "normalize con mean/std por época [N, 8]")
    for bad_std in (np.where(np.arange(N_CH) == 2, 0.0, s), np.where(np.arange(N_CH) == 2, np.nan, s)):
        try:
            normalize(ep, m, bad_std)
            raise SelftestFailure("normalize aceptó una std inválida")
        except ValueError:
            pass
    steps.append("normalize: (e - mean) / std en float32, estadísticas [8] y [N, 8], rechaza std 0/NaN")

    # quantize: mitad al par, saturación y escala/zp aleatorios contra round() de Python
    scale, zp = 0.0625, -3  # potencia de 2: z / scale exacto -> empates .5 reales
    ties = (np.arange(-140, 140, dtype=np.float32) + np.float32(0.5)) * np.float32(scale)
    q = quantize(ties, scale, zp)
    for v, qv in zip(ties, q):
        ref = max(-128, min(127, round(float(v) / scale) + zp))  # round(): mitad al par
        _check(int(qv) == ref, f"quantize({v}) = {qv}, se esperaba {ref} (mitad al par + saturación)")
    _check(int(quantize(np.float32(0.5 * scale), scale, 0)) == 0
           and int(quantize(np.float32(1.5 * scale), scale, 0)) == 2
           and int(quantize(np.float32(-0.5 * scale), scale, 0)) == 0, "quantize: empates 0.5 -> 0, 1.5 -> 2")
    _check(int(quantize(np.float32(1e30), 0.03, 5)) == 127 and int(quantize(np.float32(-1e30), 0.03, 5)) == -128,
           "quantize: saturación con valores enormes")
    for _ in range(20):
        sc = float(np.float32(rng.uniform(0.005, 0.2)))
        zpr = int(rng.integers(-128, 128))
        zz = (rng.normal(0, 3, 500) * rng.choice([1.0, 10.0], 500)).astype(np.float32)
        qq = quantize(zz, sc, zpr)
        ref = np.array([max(-128, min(127, round(float(np.float32(v) / np.float32(sc))) + zpr)) for v in zz])
        _check(qq.dtype == np.int8 and np.array_equal(qq.astype(np.int64), ref), "quantize != round() + clamp")
    for bad in ((np.array([np.nan]), 0.1, 0), (np.array([np.inf]), 0.1, 0), (np.zeros(3), 0.0, 0),
                (np.zeros(3), -0.1, 0), (np.zeros(3), 0.1, 128), (np.zeros(3), 0.1, 1.5)):
        try:
            quantize(*bad)
            raise SelftestFailure(f"quantize aceptó una entrada inválida: {bad}")
        except ValueError:
            pass
    steps.append("quantize: mitad al par (empates exactos), saturación [-128, 127], 20 escalas/zp aleatorios "
                 "== round() + clamp; rechaza no finitos, escala <= 0 y zp fuera de rango")

    # dequantize
    qa = np.arange(-128, 128, dtype=np.int8)
    for sc, zpr in ((0.0625, -3), (float(np.float32(0.0371)), 17)):
        ref = np.array([np.float32(np.float32(sc) * np.float32(int(v) - zpr)) for v in qa], dtype=np.float32)
        _check(np.array_equal(dequantize(qa, sc, zpr), ref), "dequantize != scale * (q - zp)")
    try:
        dequantize(np.array([0.5]), 0.1, 0)
        raise SelftestFailure("dequantize aceptó q no entero")
    except ValueError:
        pass
    steps.append("dequantize: scale * (float)(q - zp) en float32 para los 256 valores int8")

    # score: bucle secuencial float32 (sin FMA), formas equivalentes
    za = rng.normal(0, 1, (9, N_CH, N_T)).astype(np.float32)
    zb = (za + rng.normal(0, 0.3, za.shape)).astype(np.float32)
    sc_vec = score(za, zb)
    for i in range(za.shape[0]):
        acc = np.float32(0.0)
        for a_v, b_v in zip(za[i].ravel(), zb[i].ravel()):
            d = np.float32(a_v - b_v)
            acc = np.float32(acc + np.float32(d * d))
        ref = np.float32(acc / np.float32(N_IN))
        _check(sc_vec[i].view(np.uint32) == ref.view(np.uint32), f"score != bucle secuencial (época {i})")
        _check(abs(float(ref) - float(np.mean((za[i].astype(np.float64) - zb[i]) ** 2))) <= 1e-5 * float(ref),
               "score lejos del MSE en float64")
    _check(np.array_equal(score(za[..., None], zb.reshape(9, N_IN)), sc_vec), "score con formas (8, 40, 1) / 320")
    _check(isinstance(score(za[0], zb[0]), np.float32), "score de una época debe ser np.float32")
    steps.append("score: bit a bit igual al bucle secuencial float32 / 320; formas [8,40], [8,40,1], [320]")

    # level: ">" estricto, nextafter, no finitos y umbrales inválidos -> 3
    t1, t2, t3 = 0.5, 0.75, 1.25
    for k, t in enumerate((t1, t2, t3), 1):
        tf32 = np.float32(t)
        _check(level(tf32, t1, t2, t3) == k - 1, f"score == T{k} debe dar {k - 1} ('>' estricto)")
        _check(level(np.nextafter(tf32, np.float32(np.inf)), t1, t2, t3) == k, f"score justo por encima de T{k}")
    _check(level(0.0, t1, t2, t3) == 0 and level(1e30, t1, t2, t3) == 3, "niveles de 0 y de un score enorme")
    _check(all(level(v, t1, t2, t3) == 3 for v in (math.nan, math.inf, -math.inf)), "no finitos -> 3")
    for bad in ((0.0, 1.0, 2.0), (1.0, 0.5, 2.0), (0.5, 1.0, math.nan), (-1.0, 1.0, 2.0), (0.5, 2.0, 1.0)):
        _check(level(0.1, *bad) == 3, f"umbrales inválidos {bad} -> 3")
    sv = rng.uniform(0, 2, 1000).astype(np.float32)
    sv[::97] = np.float32(t2)
    naive = [3 if not math.isfinite(v) else 3 if v > np.float32(t3) else 2 if v > np.float32(t2)
             else 1 if v > np.float32(t1) else 0 for v in sv]
    _check(np.array_equal(level(sv, t1, t2, t3), np.array(naive)), "level vectorizado != regla escalar")
    _check(level(np.float32(0.7), 0.5, 0.75, 0.75) == 1 and level(np.float32(0.8), 0.5, 0.75, 0.75) == 3,
           "umbrales iguales (banda vacía)")
    steps.append("level: '>' estricto (score == T -> nivel inferior, nextafter -> superior), no finitos y "
                 "umbrales inválidos -> 3, vectorizado == escalar")

    # thresholds_from_scores contra np.percentile(method="linear")
    mismatches = 0
    trials = 0
    for n in (20, 21, 33, 60, 100, 257, 1000):
        for _ in range(40):
            sc_s = (rng.gamma(2.0, 0.3, n) * rng.choice([1.0, 1e-3, 1e3])).astype(np.float32)
            got_t = thresholds_from_scores(sc_s)
            ref_t = [float(np.float32(np.percentile(sc_s.astype(np.float64), p, method="linear")))
                     for p in DEFAULT_PERCENTILES]
            trials += 1
            mismatches += int(tuple(got_t) != tuple(ref_t))
            ref_np = np.percentile(sc_s.astype(np.float64), DEFAULT_PERCENTILES, method="linear")
            _check(np.allclose(got_t, ref_np, rtol=2e-7, atol=0), f"thresholds lejos de np.percentile (n={n})")
    _check(mismatches == 0, f"thresholds_from_scores != np.percentile en {mismatches} de {trials} casos")
    hand = thresholds_from_scores(np.arange(1, 21, dtype=np.float32), (50.0, 95.0, 100.0))
    _check(hand == (10.5, 19.049999237060547, 20.0), f"thresholds: caso a mano (n = 20): {hand}")
    for bad_s, bad_p in ((np.ones(19), DEFAULT_PERCENTILES), (np.r_[np.ones(25), np.nan], DEFAULT_PERCENTILES),
                         (np.zeros(30), DEFAULT_PERCENTILES), (np.ones(30), (97.0, 90.0, 99.0)),
                         (np.ones(30), (90.0, 97.0, 101.0))):
        try:
            thresholds_from_scores(bad_s, bad_p)
            raise SelftestFailure(f"thresholds_from_scores aceptó una entrada inválida "
                                  f"({bad_s.size} scores, {bad_p})")
        except ValueError:
            pass
    steps.append(f"thresholds_from_scores: == float32(np.percentile linear) en {trials} casos (n = 20..1000); "
                 "rechaza n < 20, no finitos, T1 = 0 y percentiles inválidos")
    _check(percentile_key(90.0) == "p90" and percentile_key(99.5) == "p99.5", "percentile_key")
    return steps


def _selftest_keras() -> list:
    steps = []
    for arch in ARCHS:
        model = build_model(arch)
        macs = count_macs(model)
        _check(macs == EXPECTED_MACS[arch], f"{arch}: {macs} MACs, se esperaban {EXPECTED_MACS[arch]}")
        bn = [ly.name for ly in model.layers if type(ly).__name__ == "BatchNormalization"]
        _check((arch == "errp_conv") == (len(bn) == 2), f"{arch}: capas BatchNormalization {bn}")
        m1 = inference_model(model)
        x = np.random.default_rng(0).normal(0, 1, (1,) + INPUT_SHAPE).astype(np.float32)
        _check(tuple(m1.input_shape) == (1,) + INPUT_SHAPE, f"{arch}: inference_model con lote {m1.input_shape}")
        _check(np.allclose(np.asarray(m1(x)), np.asarray(model(x, training=False)), atol=1e-6),
               f"{arch}: inference_model no reproduce el modelo")
        steps.append(f"{arch}: {model.count_params()} parámetros, {macs} MACs, (8, 40, 1) -> (8, 40, 1), "
                     "inference_model con lote 1 equivalente")
    return steps


def selftest(with_keras: bool = False) -> int:
    try:
        steps = _selftest_numpy()
        if with_keras:
            steps += _selftest_keras()
    except (SelftestFailure, ValueError, RuntimeError, ImportError) as e:
        print(f"selftest FALLÓ: {e}", file=sys.stderr)
        return 1
    for step in steps:
        print(f"  ok  {step}")
    print("selftest OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # UTF-8 también al redirigir a un pipe en Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="comprueba las funciones de referencia (solo numpy)")
    ap.add_argument("--keras", action="store_true", help="con --selftest: construye también los dos modelos")
    ap.add_argument("--summary", choices=ARCHS, help="imprime model.summary() de una arquitectura")
    args = ap.parse_args(argv)
    if args.summary:
        model = build_model(args.summary)
        model.summary()
        print(f"parámetros: {model.count_params()}  MACs: {count_macs(model)}")
        return 0
    if args.selftest:
        return selftest(with_keras=args.keras)
    ap.error("indicar --selftest [--keras] o --summary ARCH")
    return 2


if __name__ == "__main__":
    sys.exit(main())
