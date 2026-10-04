"""Preprocesamiento offline de epochs ErrP del Detector (C2).

Es la referencia que el firmware del S3 replica paso a paso (CLAUDE.md,
"Signal chain"). Cualquier cambio aquí se cambia también en el S3 y se avisa a
C1–C5.

Formato de dataset (.npz) que entrega C4 (y genera tools/make_synthetic.py):
    X        float32 [n, 8, 260]  EEG en µV, YA filtrado con el IIR causal de C1
                                  (butter(2, [1, 15], 'band', fs=250, sos), sosfilt).
                                  Ventana [-220, +820) ms alrededor del pulso de
                                  sincronía: la de [-200, +800) más 5 muestras
                                  (20 ms) de margen a cada lado para el jitter.
                                  t = 0 está en el índice T0_INDEX = 55.
    y        int8    [n]          1 = acción con error deliberado, 0 = correcta
    session  int32   [n]          id de sesión (orden cronológico dentro de cada una)
    rejected bool    [n]          opcional: epoch con HELD/GAP/SETTLING/OVERLAP
    onset_s  float64 [n]          opcional: tiempo del pulso en s (para tasas por hora)
Canales en el orden del Unicorn: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
"""
from pathlib import Path

import numpy as np

FS = 250
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
N_CH = len(CHANNELS)

PRE = 50          # [-200, 0) ms: baseline
POST = 200        # [0, 800) ms: entrada del modelo
MARGIN = 5        # ±20 ms de jitter
STORED_LEN = MARGIN + PRE + POST + MARGIN  # 260
T0_INDEX = MARGIN + PRE                    # 55
DECIM = 5         # media de cada 5 muestras: 250 -> 50 Hz
N_T = POST // DECIM                        # 40

GATE_UV = 100.0   # TUNE: pico |µV| tras el IIR en cualquier canal
FLAT_STD_UV = 0.1  # canal plano (electrodo suelto)

FILTER_SOS_DESIGN = "scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos'), sosfilt causal"


def load_epochs(path: Path) -> dict:
    d = dict(np.load(path))
    x = d["X"].astype(np.float32)
    if x.ndim != 3 or x.shape[1:] != (N_CH, STORED_LEN):
        raise SystemExit(f"{path}: X debe ser [n, {N_CH}, {STORED_LEN}], es {x.shape}")
    n = len(x)
    d["X"] = x
    d["y"] = d["y"].astype(np.int8)
    d["session"] = d.get("session", np.zeros(n)).astype(np.int32)
    d["rejected"] = d.get("rejected", np.zeros(n, bool)).astype(bool)
    return d


def preprocess(x: np.ndarray, shift: np.ndarray | int = 0) -> np.ndarray:
    """[n, 8, 260] µV -> [n, 8, 40]: baseline [-200, 0) ms restado por canal y
    media de cada 5 muestras de [0, 800) ms. shift (muestras, |shift| <= 5)
    mueve t = 0 por epoch (augmentation de jitter); 0 en vivo."""
    n = len(x)
    shifts = np.broadcast_to(np.asarray(shift, dtype=int), (n,))
    if np.abs(shifts).max(initial=0) > MARGIN:
        raise ValueError("shift fuera del margen de ±5 muestras")
    out = np.empty((n, N_CH, N_T), np.float32)
    for s in np.unique(shifts):
        m = shifts == s
        i0 = T0_INDEX + s
        base = x[m, :, i0 - PRE:i0].mean(axis=-1, keepdims=True)
        post = x[m, :, i0:i0 + POST] - base
        out[m] = post.reshape(-1, N_CH, N_T, DECIM).mean(axis=-1)
    return out


def baseline_corrected(x: np.ndarray) -> np.ndarray:
    """[n, 8, 260] -> [n, 8, 200] a 250 Hz, [0, 800) ms menos baseline (para el LDA)."""
    base = x[:, :, T0_INDEX - PRE:T0_INDEX].mean(axis=-1, keepdims=True)
    return x[:, :, T0_INDEX:T0_INDEX + POST] - base


def gate(d: dict) -> np.ndarray:
    """Máscara de epochs aceptados por el artifact gate (sin gyro: no viene en el dataset)."""
    win = d["X"][:, :, T0_INDEX - PRE:T0_INDEX + POST]
    peak_ok = np.abs(win).max(axis=(1, 2)) <= GATE_UV
    flat_ok = win.std(axis=-1).min(axis=1) >= FLAT_STD_UV
    return peak_ok & flat_ok & ~d["rejected"]


def channel_stats(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean/std por canal sobre epochs y tiempo de [n, 8, 40] (bloque de calibración)."""
    mean = e.mean(axis=(0, 2))
    std = e.std(axis=(0, 2))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(e: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((e - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def chrono_split(n: int, fractions=(0.70, 0.15, 0.15)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices train / val / test en orden cronológico (sin barajar: las acciones
    vecinas comparten estado del operador y contacto de electrodos)."""
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def augment(x: np.ndarray, rng: np.random.Generator, mean: np.ndarray, std: np.ndarray,
            noise_std: float = 0.1) -> np.ndarray:
    """Jitter ±20 ms, ganancia por canal ±10 % y ruido gaussiano (en unidades z)."""
    n = len(x)
    gain = rng.uniform(0.9, 1.1, size=(n, N_CH, 1)).astype(np.float32)
    shift = rng.integers(-MARGIN, MARGIN + 1, size=n)
    e = normalize(preprocess(x * gain, shift), mean, std)
    return e + rng.normal(0.0, noise_std, size=e.shape).astype(np.float32)


def to_model(e: np.ndarray) -> np.ndarray:
    """[n, 8, 40] -> [n, 8, 40, 1] (NHWC: alto = electrodos, ancho = tiempo)."""
    return e[..., None]


def mse_score(e: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """MSE por epoch sobre todos los ejes salvo el batch ([n, 8, 40] o [n, 8, 40, 1])."""
    return ((recon.reshape(e.shape) - e) ** 2).reshape(len(e), -1).mean(axis=1)
