#!/usr/bin/env python3
"""
Entrena el ErrP-AE del detector C2 (INTUNE) con la receta del CLAUDE.md del equipo y
deja en --out DIR lo que necesita el exportador a C (ml/c_exporter/export_to_c.py).

Receta:
  - solo épocas tras acciones CORRECTAS (deliberate_error = false); las de error se usan
    únicamente para evaluar;
  - split entrenamiento / held-out de épocas correctas: por sesión si hay >= 3 sesiones
    (sesiones completas fuera, como un operador/sesión nuevo), si no aleatorio;
  - normalización z-score por canal: por sesión (estadísticas de las épocas correctas de
    entrenamiento de esa sesión; en una sesión held-out, de sus épocas correctas, como la
    calibración del S3) si los datos traen "session"; si no, agregadas (pooled). Las
    agregadas de entrenamiento son los valores por defecto de config.json;
  - aumento de datos en cada pasada: jitter +-20 ms (desplazamiento de hasta 5 muestras a
    250 Hz con ventanas crudas; con épocas ya preprocesadas, +-1 muestra a 50 Hz),
    ganancia por canal U(0.9, 1.1) y ruido gaussiano pequeño (desv. 0.05 en unidades z,
    solo en la entrada: el objetivo es la época sin ruido);
  - pérdida MSE, Adam + weight decay L2 en los kernels, parada temprana por el MSE de las
    épocas correctas held-out (se restauran los mejores pesos);
  - umbrales por defecto = percentiles 90/97/99 del score de las correctas held-out.

Datos de entrada (formato PROVISIONAL hasta que C4 fije el suyo), un .npz con:
  "windows"  [N, 8, 250] float  ventanas crudas [-200, +800) ms a 250 Hz tras el IIR causal (uV),
                                canales Fz, C3, Cz, C4, Pz, PO7, Oz, PO8; se preprocesan con
                                errp_ae.preprocess_window (= ae_preprocess del S3); o bien
  "epochs"   [N, 8, 40] float   épocas ya preprocesadas (línea base restada, 50 Hz, uV).
                                Si vienen las dos se usa "windows" (jitter a 250 Hz).
  "is_error" [N] bool (o 0/1)   True = acción con error deliberado (deliberate_error).
  "session"  [N] int, opcional  sesión u operador de cada época.
  Las épocas con valores no finitos se descartan con un aviso.
--synthetic genera en su lugar un conjunto sintético reproducible (ruido de fondo en uV con
offset DC; las épocas de error llevan una deflexión tipo ErrP en Fz/Cz: negatividad ~250 ms y
positividad ~400 ms). Un modelo entrenado así es un PLACEHOLDER ("placeholder": true).

Salidas en --out DIR (se sobrescriben; cada archivo se escribe de forma atómica):
  model.keras  modelo Keras 3 con los mejores pesos (carga sin objetos personalizados:
               keras.models.load_model(ruta, compile=False))
  config.json  {"arch", "channels", "fs_hz": 250, "decim": 5, "epoch_ms": [-200, 800],
                "baseline_ms": [-200, 0], "norm": {"mean": [8], "std": [8]},
                "percentiles": [90, 97, 99], "thresholds": {"p90": T1, "p97": T2, "p99": T3},
                "score": "mse", "placeholder": bool,
                "_comentario" y "training" (informativos: el exportador los ignora)}
               Las claves de "thresholds" salen de errp_ae.percentile_key (p99.5 -> "p99.5").
  rep.npz      "z" [n, 8, 40] float32: épocas correctas de ENTRENAMIENTO normalizadas, sin aumento
               (hasta --n-rep), conjunto representativo de la PTQ int8.
  val.npz      épocas correctas held-out y TODAS las de error (nunca se entrenan con ellas):
               "z" [M, 8, 40] float32 normalizadas (entrada del modelo), "is_error" [M] bool,
               "epochs" [M, 8, 40] float32 (uV, antes de normalizar), "mean"/"std" [M, 8] float32
               (estadísticas con que se normalizó cada época: z = normalize(epochs, mean, std)),
               "score" [M] float32 (score del modelo float) y "session" [M] int64 si hay sesiones.
  rep.npz y val.npz son deterministas byte a byte; model.keras no (Keras guarda la fecha).

Determinismo: misma máquina, semilla y datos -> mismos pesos y archivos. Antes de importar
TensorFlow el script fija el proceso a una CPU (en CPUs híbridas P/E el blocking de las GEMM de
Eigen depende del tipo de núcleo) y TF_ENABLE_ONEDNN_OPTS=0 por defecto; después, un hilo
intra/inter-op y op determinism.

Uso:
  python ml/autoencoder/train_errp_ae.py --data datos.npz --out runs/r1
  python ml/autoencoder/train_errp_ae.py --synthetic --out /tmp/ae_synth [--arch dense]
Después (exportar a C):
  python ml/c_exporter/export_to_c.py --model DIR/model.keras --config DIR/config.json \\
         --rep DIR/rep.npz --val DIR/val.npz
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import errp_ae as ae  # noqa: E402

SCRIPT_REL = "ml/autoencoder/train_errp_ae.py"
EXPORTER_REL = "ml/c_exporter/export_to_c.py"
MIN_SESSION_EPOCHS = 10     # mínimo de épocas correctas para estadísticas propias de una sesión
MIN_TRAIN_EPOCHS = 20       # mínimo de épocas correctas de entrenamiento
PRINT_EVERY = 10            # líneas de progreso: cada N épocas y en cada mejora


class TrainError(Exception):
    """Problema esperado (datos inválidos, pocos datos...): mensaje claro y exit 1."""


def warn(msg: str) -> None:
    print(f"AVISO: {msg}", file=sys.stderr)


def f32(value: float) -> float:
    """Float de Python con el texto más corto que vuelve al mismo float32 (JSON)."""
    return float(str(np.float32(value)))


def json_number(value: float):
    """Entero si es entero (90 -> 90, como en el contrato), si no float32 más corto."""
    v = float(value)
    return int(v) if v.is_integer() else f32(v)


def write_atomic(path: Path, data: bytes) -> None:
    """Escritura atómica: temporal en el mismo directorio y os.replace."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    except OSError as e:
        raise TrainError(f"no se puede escribir en {path.parent}: {e}") from None
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp crea con 0600
        os.replace(tmp, str(path))
    except BaseException as e:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        if isinstance(e, OSError):
            raise TrainError(f"no se pudo escribir {path}: {e}") from None
        raise


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """.npz (zip sin compresión) determinista: fecha fija en las entradas (np.savez pone la hora
    actual) y sin pickle. Se lee con np.load."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            one = io.BytesIO()
            np.lib.format.write_array(one, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, one.getvalue())
    return buf.getvalue()


# ---- Datos ----

@dataclass
class Dataset:
    epochs: np.ndarray                  # [N, 8, 40] float32, uV preprocesadas
    windows: Optional[np.ndarray]       # [N, 8, 250] float32 o None
    is_error: np.ndarray                # [N] bool
    session: Optional[np.ndarray]       # [N] int64 o None
    source: str                         # descripción (sin rutas absolutas)


def _gauss(t: np.ndarray, center: np.ndarray, width: float) -> np.ndarray:
    """Pulsos gaussianos [n, T] centrados en center [n] (segundos)."""
    return np.exp(-0.5 * ((t[None, :] - center[:, None]) / width) ** 2)


# Topografías (orden Fz, C3, Cz, C4, Pz, PO7, Oz, PO8) del generador sintético
SYN_ERRP_TOPO = np.array([1.0, 0.45, 1.0, 0.45, 0.5, 0.15, 0.1, 0.15])   # fronto-central (FCz)
SYN_ERP_TOPO = np.array([0.3, 0.5, 0.8, 0.5, 1.0, 0.6, 0.5, 0.6])       # positividad centro-parietal
SYN_ALPHA_TOPO = np.array([0.2, 0.3, 0.3, 0.3, 0.6, 1.0, 1.0, 1.0])     # alfa occipital
SYN_BG_SOURCES, SYN_BG_AR, SYN_BG_UV = 8, 0.96, 6.0   # fuentes de fondo AR(1): número, coeficiente, uV
SYN_NE = (-8.0, 0.250, 0.035)   # ErrP, negatividad: amplitud (uV), latencia (s), anchura (s)
SYN_PE = (10.0, 0.400, 0.060)   # ErrP, positividad


def make_synthetic(seed: int = 0, n_sessions: int = 3, n_correct: int = 200,
                   n_error: int = 40) -> Dict[str, np.ndarray]:
    """Conjunto sintético reproducible en el formato de entrada ("windows", "is_error",
    "session"), en uV a 250 Hz, ventanas de [-200, +800) ms:
      fondo: 8 fuentes AR(1) (espectro ~1/f, ~6 uV) mezcladas a los 8 canales (rango
             completo) con una mezcla por sesión, alfa occipital 8.5-11.5 Hz, ruido de sensor,
             deriva lenta y offset DC por canal (sesión + época); ganancia por canal y
             sesión U(0.7, 1.3);
      todas: positividad centro-parietal ~300 ms (respuesta a la acción, 1.5-4 uV);
      error: ErrP en Fz/Cz: negatividad ~250 ms (-8 uV) y positividad ~400 ms (+10 uV), con
             amplitud x U(0.7, 1.3) y latencia +-30 ms por época.
    La separación error/correcta que se obtiene NO predice el rendimiento con EEG real."""
    if n_sessions < 1 or n_correct < 0 or n_error < 0 or n_correct + n_error < 1:
        raise TrainError("parámetros del conjunto sintético no válidos")
    rng = np.random.default_rng([seed, 1000])
    t = (np.arange(ae.WIN) - ae.PRE) / float(ae.FS_HZ)  # s, de -0.2 a 0.796
    k_src, a_ar = SYN_BG_SOURCES, SYN_BG_AR
    base_mix = rng.normal(0.0, 1.0, (ae.N_CH, k_src)) / math.sqrt(k_src)
    windows, is_error, session = [], [], []
    for s in range(n_sessions):
        n = n_correct + n_error
        err = np.zeros(n, dtype=bool)
        err[:n_error] = True
        rng.shuffle(err)
        mix = base_mix + rng.normal(0.0, 0.25, base_mix.shape) / math.sqrt(k_src)
        gain = rng.uniform(0.7, 1.3, ae.N_CH)
        dc = rng.uniform(-40.0, 40.0, ae.N_CH)
        white = rng.normal(0.0, 1.0, (n, k_src, ae.WIN))
        src = np.empty_like(white)
        src[..., 0] = white[..., 0] / math.sqrt(1.0 - a_ar * a_ar)
        for j in range(1, ae.WIN):
            src[..., j] = a_ar * src[..., j - 1] + white[..., j]
        src *= SYN_BG_UV * math.sqrt(1.0 - a_ar * a_ar)  # desviación estacionaria SYN_BG_UV por fuente
        x = np.einsum("ck,nkt->nct", mix, src)
        f_a = rng.uniform(8.5, 11.5, n)
        ph_a = rng.uniform(0.0, 2.0 * math.pi, n)
        amp_a = rng.uniform(1.0, 5.0, n)
        alpha = amp_a[:, None] * np.sin(2.0 * math.pi * f_a[:, None] * t[None, :] + ph_a[:, None])
        x += SYN_ALPHA_TOPO[None, :, None] * alpha[:, None, :]
        erp = rng.uniform(1.5, 4.0, n)[:, None] * _gauss(t, 0.30 + rng.uniform(-0.03, 0.03, n), 0.06)
        x += SYN_ERP_TOPO[None, :, None] * erp[:, None, :]
        lat = rng.uniform(-0.03, 0.03, n)
        amp = rng.uniform(0.7, 1.3, n)
        errp = amp[:, None] * (SYN_NE[0] * _gauss(t, SYN_NE[1] + lat, SYN_NE[2])
                               + SYN_PE[0] * _gauss(t, SYN_PE[1] + lat, SYN_PE[2]))
        x[err] += SYN_ERRP_TOPO[None, :, None] * errp[err][:, None, :]
        drift = rng.normal(0.0, 4.0, (n, ae.N_CH, 1)) * t[None, None, :]  # uV/s
        x = gain[None, :, None] * (x + drift) + rng.normal(0.0, 1.5, x.shape)
        x += dc[None, :, None] + rng.normal(0.0, 3.0, (n, ae.N_CH, 1))
        windows.append(x.astype(np.float32))
        is_error.append(err)
        session.append(np.full(n, s, dtype=np.int64))
    return {"windows": np.concatenate(windows), "is_error": np.concatenate(is_error),
            "session": np.concatenate(session)}


def dataset_from_arrays(arrays: Mapping[str, np.ndarray], source: str) -> Dataset:
    """Valida el formato de entrada y preprocesa las ventanas si las hay."""
    keys = set(arrays.keys())
    if "is_error" not in keys or not ({"windows", "epochs"} & keys):
        raise TrainError(f"{source}: faltan claves; se esperaba 'is_error' y 'windows' o 'epochs' "
                         f"(hay: {sorted(keys)})")
    windows = None
    if "windows" in keys:
        windows = np.asarray(arrays["windows"])
        if windows.ndim != 3 or windows.shape[1:] != (ae.N_CH, ae.WIN):
            raise TrainError(f"{source}: 'windows' con forma {windows.shape}; se esperaba [N, 8, 250]")
        if not np.issubdtype(windows.dtype, np.floating) and not np.issubdtype(windows.dtype, np.integer):
            raise TrainError(f"{source}: 'windows' no es numérico ({windows.dtype})")
        with np.errstate(over="ignore"):
            windows = windows.astype(np.float32)
        n = windows.shape[0]
        if "epochs" in keys:
            print("  nota: el .npz trae 'windows' y 'epochs'; se usan 'windows' (preprocesadas aquí)")
    else:
        ep = np.asarray(arrays["epochs"])
        if ep.ndim != 3 or ep.shape[1:] != (ae.N_CH, ae.N_T):
            raise TrainError(f"{source}: 'epochs' con forma {ep.shape}; se esperaba [N, 8, 40]")
        if not np.issubdtype(ep.dtype, np.floating) and not np.issubdtype(ep.dtype, np.integer):
            raise TrainError(f"{source}: 'epochs' no es numérico ({ep.dtype})")
        n = ep.shape[0]
    if n < 1:
        raise TrainError(f"{source}: no hay épocas")
    is_error = np.asarray(arrays["is_error"])
    if is_error.shape != (n,):
        raise TrainError(f"{source}: 'is_error' con forma {is_error.shape}; se esperaba ({n},)")
    if is_error.dtype != np.bool_:
        if not np.issubdtype(is_error.dtype, np.integer) or not np.all((is_error == 0) | (is_error == 1)):
            raise TrainError(f"{source}: 'is_error' debe ser bool (o enteros 0/1)")
        is_error = is_error.astype(bool)
    session = None
    if "session" in keys:
        session = np.asarray(arrays["session"])
        if session.shape != (n,) or not np.issubdtype(session.dtype, np.integer):
            raise TrainError(f"{source}: 'session' debe ser un vector de enteros de forma ({n},)")
        session = session.astype(np.int64)
    if windows is not None:
        epochs = ae.preprocess_window(windows)
        finite = np.all(np.isfinite(windows), axis=(1, 2))
    else:
        with np.errstate(over="ignore"):
            epochs = np.asarray(arrays["epochs"]).astype(np.float32)
        finite = np.all(np.isfinite(epochs), axis=(1, 2))
    if not np.all(finite):
        warn(f"{source}: {int(np.sum(~finite))} épocas con valores no finitos descartadas")
        keep = np.flatnonzero(finite)
        epochs = epochs[keep]
        windows = windows[keep] if windows is not None else None
        is_error = is_error[keep]
        session = session[keep] if session is not None else None
    return Dataset(np.ascontiguousarray(epochs), windows, is_error, session, source)


def load_npz(path: Path) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise TrainError(f"no existe el archivo de datos: {path}")
    if not zipfile.is_zipfile(path):
        raise TrainError(f"{path.name} no es un .npz (np.savez): formato de entrada arriba (--help)")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        raise TrainError(f"no se pudo leer {path.name} como .npz: {e}") from None
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return dataset_from_arrays(arrays, f"{path.name} (sha256 {sha})")


# ---- Split y normalización ----

@dataclass
class Split:
    train: np.ndarray                   # índices de épocas correctas de entrenamiento
    heldout: np.ndarray                 # índices de épocas correctas held-out
    mode: str                           # "session" | "random"
    heldout_sessions: List[int] = field(default_factory=list)


def split_correct(ds: Dataset, val_frac: float, mode: str, rng: np.random.Generator) -> Split:
    """Split de las épocas correctas. auto: por sesión con >= 3 sesiones, si no aleatorio."""
    correct = np.flatnonzero(~ds.is_error)
    n_min = MIN_TRAIN_EPOCHS + ae.CALIB_MIN_SCORES
    if correct.size < n_min:
        raise TrainError(f"hay {correct.size} épocas correctas; hacen falta al menos {n_min} "
                         f"({MIN_TRAIN_EPOCHS} de entrenamiento + {ae.CALIB_MIN_SCORES} held-out para los umbrales)")
    sessions = np.unique(ds.session[correct]) if ds.session is not None else np.array([], dtype=np.int64)
    if mode == "session" and sessions.size < 2:
        raise TrainError("--split session requiere 'session' con >= 2 sesiones con épocas correctas")
    if mode == "session" or (mode == "auto" and sessions.size >= 3):
        assert ds.session is not None
        counts = {int(s): int(np.sum(ds.session[correct] == s)) for s in sessions}
        target = max(val_frac * correct.size, float(ae.CALIB_MIN_SCORES))
        held: List[int] = []
        n_held = 0
        for s in rng.permutation(sessions):
            if n_held >= target or len(held) >= sessions.size - 1:
                break
            held.append(int(s))
            n_held += counts[int(s)]
        in_held = np.isin(ds.session[correct], held)
        sp = Split(correct[~in_held], correct[in_held], "session", sorted(held))
        if sp.heldout.size >= ae.CALIB_MIN_SCORES and sp.train.size >= MIN_TRAIN_EPOCHS:
            return sp
        if mode == "session":
            raise TrainError(f"split por sesión: {sp.heldout.size} held-out / {sp.train.size} de entrenamiento; "
                             f"hacen falta >= {ae.CALIB_MIN_SCORES} / {MIN_TRAIN_EPOCHS}")
        warn("split por sesión imposible (sesiones demasiado pequeñas): se usa un split aleatorio")
    perm = rng.permutation(correct)
    n_val = min(max(ae.CALIB_MIN_SCORES, int(round(val_frac * correct.size))), correct.size - MIN_TRAIN_EPOCHS)
    return Split(np.sort(perm[n_val:]), np.sort(perm[:n_val]), "random")


@dataclass
class Norm:
    mean: np.ndarray                    # [N, 8] float32: estadísticas con que se normaliza cada época
    std: np.ndarray                     # [N, 8] float32
    pooled_mean: np.ndarray             # [8]: agregadas del entrenamiento (config.json "norm")
    pooled_std: np.ndarray              # [8]
    mode: str                           # "session" | "pooled"
    origin: Dict[int, str] = field(default_factory=dict)  # sesión -> de dónde salen sus estadísticas


def _stats(ds: Dataset, idx: np.ndarray, what: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return ae.norm_stats(ds.epochs[idx])
    except ValueError as e:
        raise TrainError(f"estadísticas de normalización de {what}: {e}") from None


def normalization(ds: Dataset, sp: Split) -> Norm:
    """Por sesión si hay "session": correctas de entrenamiento de la sesión, o sus correctas
    held-out si es una sesión held-out (calibración del S3); si no, las agregadas."""
    pm, ps = _stats(ds, sp.train, "las épocas correctas de entrenamiento")
    n = ds.epochs.shape[0]
    mean = np.repeat(pm[None, :], n, axis=0)
    std = np.repeat(ps[None, :], n, axis=0)
    if ds.session is None:
        return Norm(mean, std, pm, ps, "pooled")
    origin: Dict[int, str] = {}
    for s in np.unique(ds.session):
        tr = sp.train[ds.session[sp.train] == s]
        ho = sp.heldout[ds.session[sp.heldout] == s]
        if tr.size >= MIN_SESSION_EPOCHS:
            idx, origin[int(s)] = tr, f"{tr.size} correctas de entrenamiento"
        elif ho.size >= MIN_SESSION_EPOCHS:
            idx, origin[int(s)] = ho, f"{ho.size} correctas held-out (calibración)"
        else:
            origin[int(s)] = "agregadas (menos de {} correctas)".format(MIN_SESSION_EPOCHS)
            warn(f"sesión {int(s)}: menos de {MIN_SESSION_EPOCHS} épocas correctas; se normaliza con las "
                 "estadísticas agregadas")
            continue
        m, sd = _stats(ds, idx, f"la sesión {int(s)}")
        sel = ds.session == s
        mean[sel] = m
        std[sel] = sd
    return Norm(mean, std, pm, ps, "session", origin)


# ---- Aumento de datos ----

@dataclass
class Augment:
    jitter_ms: float = 20.0
    noise_std: float = 0.05
    gain: float = 0.1


def shift_edge(a: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """out[i, c, t] = a[i, c, clip(t + shifts[i], 0, T - 1)] (desplazamiento con réplica del borde)."""
    n, n_ch, n_t = a.shape
    t_idx = np.clip(np.arange(n_t)[None, :] + shifts[:, None], 0, n_t - 1)
    return a[np.arange(n)[:, None, None], np.arange(n_ch)[None, :, None], t_idx[:, None, :]]


def jitter_samples(aug: Augment, windows: bool) -> int:
    """Desplazamiento máximo en muestras: 20 ms -> 5 a 250 Hz (ventanas) o 1 a 50 Hz (épocas)."""
    fs = ae.FS_HZ if windows else ae.FS_HZ / ae.DECIM
    return int(round(aug.jitter_ms * fs / 1000.0))


def augmented(ds: Dataset, idx: np.ndarray, norm: Norm, aug: Augment,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Una pasada de aumento: (entrada con ruido, objetivo sin ruido), ambos z [n, 8, 40]."""
    n = idx.size
    jmax = jitter_samples(aug, ds.windows is not None)
    shifts = rng.integers(-jmax, jmax + 1, size=n) if jmax > 0 else np.zeros(n, dtype=np.int64)
    if ds.windows is not None:
        e = ae.preprocess_window(shift_edge(ds.windows[idx], shifts))
    else:
        e = shift_edge(ds.epochs[idx], shifts)
    if aug.gain > 0:
        e = e * rng.uniform(1.0 - aug.gain, 1.0 + aug.gain, (n, ae.N_CH, 1)).astype(np.float32)
    y = ae.normalize(e, norm.mean[idx], norm.std[idx])
    x = y + rng.normal(0.0, aug.noise_std, y.shape).astype(np.float32) if aug.noise_std > 0 else y
    return x.astype(np.float32, copy=False), y


# ---- Entrenamiento (TensorFlow solo aquí) ----

def import_tf():
    """TensorFlow + Keras deterministas: proceso fijado a una CPU antes de importar TensorFlow
    (errp_ae.pin_single_cpu: en CPUs híbridas P/E el blocking de las GEMM de Eigen depende del
    tipo de núcleo), TF_ENABLE_ONEDNN_OPTS=0 (por defecto) y UN hilo intra/inter-op (con varios,
    Eigen reparte algunas contracciones por la dimensión interna y suma los parciales en el orden
    en que acaban los hilos). Con este modelo tan pequeño un hilo apenas cuesta."""
    if "tensorflow" not in sys.modules and not ae.pin_single_cpu():
        warn("no se pudo fijar el proceso a una CPU: en CPUs híbridas el entrenamiento puede no ser "
             "determinista en los últimos bits")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        keras = ae.keras_module()
        import tensorflow as tf
    except (ImportError, RuntimeError) as e:
        raise TrainError(str(e)) from None
    if os.environ.get("TF_ENABLE_ONEDNN_OPTS") != "0":
        warn("TF_ENABLE_ONEDNN_OPTS no es 0: oneDNN puede cambiar los resultados entre ejecuciones")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:  # el runtime de TensorFlow ya estaba inicializado en este proceso
        warn("no se pudo fijar un solo hilo en TensorFlow: el entrenamiento puede no ser determinista")
    return tf, keras


@dataclass
class TrainResult:
    model: object
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    best_val_mse: float
    seconds: float


PREDICT_BATCH = 1024


def predict_scores(model, z: np.ndarray) -> np.ndarray:
    """Score float32 (errp_ae.score) del modelo Keras en modo inferencia (predict_on_batch: sin
    la canalización tf.data de predict())."""
    out = [np.zeros(0, dtype=np.float32)]
    for b in range(0, z.shape[0], PREDICT_BATCH):
        zb = z[b:b + PREDICT_BATCH]
        out.append(np.atleast_1d(ae.score(zb, np.asarray(model.predict_on_batch(zb[..., None])))))
    return np.concatenate(out)


def train(ds: Dataset, sp: Split, norm: Norm, z_val: np.ndarray, args: argparse.Namespace) -> TrainResult:
    """Bucle propio: una pasada de aumento por época, lotes con train_on_batch en un orden
    barajado con la semilla, MSE held-out y parada temprana con restauración de los mejores pesos."""
    tf, keras = import_tf()
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng([args.seed, 2])
    aug = Augment(args.jitter_ms, args.noise_std, args.gain)
    model = ae.build_model(args.arch, args.l2)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss="mse")
    best_val, best_epoch, best_weights = math.inf, 0, model.get_weights()
    t0 = time.perf_counter()
    epoch = 0
    stopped = False
    bs = args.batch_size
    print(f"Entrenando {args.arch} ({model.count_params()} parámetros): {sp.train.size} épocas correctas, "
          f"held-out {sp.heldout.size}, lote {bs}, máx. {args.epochs} épocas, paciencia {args.patience}")
    for epoch in range(1, args.epochs + 1):
        x, y = augmented(ds, sp.train, norm, aug, rng)
        perm = rng.permutation(x.shape[0])
        x, y = x[perm][..., None], y[perm][..., None]
        model.reset_metrics()  # la pérdida devuelta es la media de la época (ponderada por lote)
        loss = math.nan
        for b in range(0, x.shape[0], bs):
            loss = float(np.asarray(model.train_on_batch(x[b:b + bs], y[b:b + bs])).ravel()[0])
        val = float(np.mean(predict_scores(model, z_val), dtype=np.float64))
        if not (math.isfinite(loss) and math.isfinite(val)):
            raise TrainError(f"época {epoch}: pérdida no finita (train {loss}, held-out {val}); bajar --lr")
        improved = val < best_val
        if improved:
            best_val, best_epoch, best_weights = val, epoch, model.get_weights()
        if improved or epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  época {epoch:4d}  pérdida train (con L2) {loss:.5f}  MSE held-out {val:.5f}"
                  f"{'  *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            stopped = True
            break
    model.set_weights(best_weights)
    seconds = time.perf_counter() - t0
    if stopped:
        print(f"  parada temprana en la época {epoch}: {args.patience} épocas sin mejorar desde la {best_epoch}")
    else:
        warn(f"no hubo parada temprana en {args.epochs} épocas (mejor: {best_epoch}); considerar subir --epochs")
    return TrainResult(model, best_epoch, epoch, stopped, best_val, seconds)


# ---- Evaluación y salidas ----

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC (Mann-Whitney) de pos > neg, empates a medias."""
    x = np.concatenate([pos, neg]).astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(1, x.size + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n_p, n_n = pos.size, neg.size
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def level_rates(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    lv = np.atleast_1d(ae.level(scores, *thresholds))
    return np.array([np.mean(lv == k) for k in range(4)]) if lv.size else np.zeros(4)


def fmt_rates(r: np.ndarray) -> str:
    return " / ".join(f"{100.0 * v:5.1f}%" for v in r)


def build_config(args: argparse.Namespace, ds: Dataset, sp: Split, norm: Norm, res: TrainResult,
                 thresholds: Tuple[float, float, float], n_rep: int, n_macs: int) -> Dict[str, object]:
    pcts = [float(p) for p in args.percentiles]
    n_err = int(np.sum(ds.is_error))
    return {
        "_comentario": ("Generado por " + SCRIPT_REL + ". norm = media/std agregadas de las épocas correctas de "
                        "entrenamiento (calibración por defecto del S3); thresholds = percentiles del score "
                        "MSE de las épocas correctas held-out. En el S3 se recalibra por operador y sesión."),
        "arch": args.arch,
        "channels": list(ae.CHANNELS),
        "fs_hz": ae.FS_HZ,
        "decim": ae.DECIM,
        "epoch_ms": list(ae.EPOCH_MS),
        "baseline_ms": list(ae.BASELINE_MS),
        "norm": {"mean": [f32(v) for v in norm.pooled_mean], "std": [f32(v) for v in norm.pooled_std]},
        "percentiles": [json_number(p) for p in pcts],
        "thresholds": {ae.percentile_key(p): f32(t) for p, t in zip(pcts, thresholds)},
        "score": "mse",
        "placeholder": bool(args.synthetic),
        "training": {
            "data": ds.source,
            "seed": args.seed,
            "n_correct_train": int(sp.train.size),
            "n_correct_heldout": int(sp.heldout.size),
            "n_error": n_err,
            "split": sp.mode,
            "heldout_sessions": sp.heldout_sessions,
            "norm": norm.mode,
            "epochs_max": args.epochs,
            "epochs_run": res.epochs_run,
            "best_epoch": res.best_epoch,
            "early_stopping": res.stopped_early,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l2": args.l2,
            "augment": {"jitter_ms": args.jitter_ms,
                        "jitter_samples": jitter_samples(Augment(args.jitter_ms), ds.windows is not None),
                        "jitter_fs_hz": ae.FS_HZ if ds.windows is not None else ae.FS_HZ // ae.DECIM,
                        "noise_std": args.noise_std, "gain": args.gain},
            "heldout_mse": f32(res.best_val_mse),
            "n_params": int(res.model.count_params()),
            "n_macs": n_macs,
            "n_rep": n_rep,
        },
    }


def write_outputs(out_dir: Path, model, arch: str, l2: float, config: Mapping[str, object],
                  rep: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> None:
    """model.keras, rep.npz y val.npz; config.json el último. model.keras es una copia SIN
    compilar (misma arquitectura y pesos, sin el estado del optimizador)."""
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise TrainError(f"no se puede crear {out_dir}: {e}") from None
    clean = ae.build_model(arch, l2)
    clean.set_weights(model.get_weights())
    tmp = out_dir / f".model.{os.getpid()}.tmp.keras"  # Keras exige la extensión .keras
    try:
        clean.save(str(tmp))
        os.replace(str(tmp), str(out_dir / "model.keras"))
    except OSError as e:
        raise TrainError(f"no se pudo escribir {out_dir / 'model.keras'}: {e}") from None
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    write_atomic(out_dir / "rep.npz", npz_bytes(rep))
    write_atomic(out_dir / "val.npz", npz_bytes(val))
    write_atomic(out_dir / "config.json",
                 (json.dumps(config, indent=2, ensure_ascii=True) + "\n").encode("ascii"))


def run(args: argparse.Namespace) -> int:
    if args.synthetic:
        arrays = make_synthetic(args.seed, args.syn_sessions, args.syn_correct, args.syn_errors)
        ds = dataset_from_arrays(arrays, f"sintéticos (semilla {args.seed}, {args.syn_sessions} sesiones x "
                                         f"({args.syn_correct} correctas + {args.syn_errors} con error))")
    else:
        ds = load_npz(args.data)
    sp = split_correct(ds, args.val_frac, args.split, np.random.default_rng([args.seed, 1]))
    norm = normalization(ds, sp)
    z_all = ae.normalize(ds.epochs, norm.mean, norm.std)
    if not np.all(np.isfinite(z_all)):
        raise TrainError("la normalización produjo valores no finitos (¿épocas enormes?)")
    z_val = z_all[sp.heldout]
    res = train(ds, sp, norm, z_val, args)
    model = res.model

    s_val = predict_scores(model, z_val)
    try:
        thresholds = ae.thresholds_from_scores(s_val, args.percentiles)
    except ValueError as e:
        raise TrainError(f"umbrales: {e}") from None
    err_idx = np.flatnonzero(ds.is_error)
    s_err = predict_scores(model, z_all[err_idx])

    rng_rep = np.random.default_rng([args.seed, 3])
    rep_idx = np.sort(rng_rep.choice(sp.train, size=min(args.n_rep, sp.train.size), replace=False))
    val_idx = np.concatenate([sp.heldout, err_idx])
    rep = {"z": z_all[rep_idx]}
    val: Dict[str, np.ndarray] = {
        "z": z_all[val_idx],
        "is_error": ds.is_error[val_idx].copy(),
        "epochs": ds.epochs[val_idx],
        "mean": norm.mean[val_idx],
        "std": norm.std[val_idx],
        "score": np.concatenate([s_val, s_err]).astype(np.float32),
    }
    if ds.session is not None:
        val["session"] = ds.session[val_idx]
    n_macs = ae.count_macs(model)
    config = build_config(args, ds, sp, norm, res, thresholds, int(rep_idx.size), n_macs)
    write_outputs(args.out, model, args.arch, args.l2, config, rep, val)

    # ---- Resumen ----
    pcts = [float(p) for p in args.percentiles]
    print("Entrenamiento ErrP-AE completado:")
    print(f"  arquitectura  : {args.arch} ({model.count_params()} parámetros, {n_macs} MACs)")
    print(f"  datos         : {ds.source}")
    print(f"                  {int(np.sum(~ds.is_error))} correctas, {err_idx.size} con error"
          f"{'' if ds.session is None else f', {np.unique(ds.session).size} sesiones'}"
          f"{'; ventanas crudas (jitter a 250 Hz)' if ds.windows is not None else '; épocas (jitter a 50 Hz)'}")
    held = f" (sesiones held-out {sp.heldout_sessions})" if sp.mode == "session" else ""
    print(f"  split         : {sp.mode}{held}: {sp.train.size} entrenamiento / {sp.heldout.size} held-out")
    print(f"  normalización : {'por sesión' if norm.mode == 'session' else 'agregada'}; config.json norm = "
          "agregadas del entrenamiento")
    for s, origin in norm.origin.items():
        print(f"                  sesión {s}: {origin}")
    print(f"  épocas        : mejor {res.best_epoch} de {res.epochs_run} "
          f"({'parada temprana' if res.stopped_early else 'SIN parada temprana'}, paciencia {args.patience}), "
          f"{res.seconds:.1f} s")
    print(f"  MSE held-out  : {res.best_val_mse:.5f} (épocas correctas held-out, n = {s_val.size})")
    print("  umbrales      : " + "  ".join(f"{ae.percentile_key(p)}={f32(t)}" for p, t in zip(pcts, thresholds)))
    print(f"  niveles 0/1/2/3 correctas held-out : {fmt_rates(level_rates(s_val, thresholds))}")
    if err_idx.size:
        print(f"  niveles 0/1/2/3 con error (n={err_idx.size:4d}) : {fmt_rates(level_rates(s_err, thresholds))}")
        msg = (f"  score medio   : correctas held-out {float(np.mean(s_val)):.4f} | "
               f"con error {float(np.mean(s_err)):.4f}")
        if sp.mode == "session" and ds.session is not None:
            in_held = np.isin(ds.session[err_idx], sp.heldout_sessions)
            if np.any(in_held):
                print(f"  niveles 0/1/2/3 con error, sesiones held-out (n={int(np.sum(in_held))}) : "
                      f"{fmt_rates(level_rates(s_err[in_held], thresholds))}")
                msg += f" (sesiones held-out {float(np.mean(s_err[in_held])):.4f})"
                print(msg)
                print(f"  AUC error vs correctas held-out: {auc(s_err, s_val):.3f} (todas) | "
                      f"{auc(s_err[in_held], s_val):.3f} (sesiones held-out)")
            else:
                print(msg)
                print(f"  AUC error vs correctas held-out: {auc(s_err, s_val):.3f}")
        else:
            print(msg)
            print(f"  AUC error vs correctas held-out: {auc(s_err, s_val):.3f}")
    print(f"  placeholder   : {'SÍ (datos sintéticos: NO usar con operadores reales)' if args.synthetic else 'no'}")
    print(f"  salida        : {args.out}")
    print(f"                  model.keras, config.json, rep.npz (z {rep['z'].shape}), "
          f"val.npz (z {val['z'].shape}, {int(np.sum(val['is_error']))} con error)")
    out = str(args.out)
    print("Siguiente paso (exportar a C):")
    print(f'  python {EXPORTER_REL} --model "{os.path.join(out, "model.keras")}" --config '
          f'"{os.path.join(out, "config.json")}" --rep "{os.path.join(out, "rep.npz")}" '
          f'--val "{os.path.join(out, "val.npz")}"')
    return 0


# ---- CLI ----

def _int_range(lo: int, hi: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"entero no válido: {text!r}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"debe estar entre {lo} y {hi}")
        return value
    return parse


def _float_range(lo: float, hi: float, lo_open: bool = False):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"número no válido: {text!r}") from None
        if not math.isfinite(value) or value > hi or value < lo or (lo_open and value == lo):
            raise argparse.ArgumentTypeError(f"debe estar en {'(' if lo_open else '['}{lo}, {hi}]")
        return value
    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help=".npz de entrada (formato arriba)")
    src.add_argument("--synthetic", action="store_true", help="conjunto sintético reproducible (placeholder)")
    ap.add_argument("--out", type=Path, required=True, help="directorio de salida")
    ap.add_argument("--arch", choices=ae.ARCHS, default="errp_conv", help="arquitectura (def. errp_conv)")
    ap.add_argument("--seed", type=_int_range(0, 2 ** 31 - 1), default=0, help="semilla (def. 0)")
    ap.add_argument("--epochs", type=_int_range(1, 100000), default=300, help="máximo de épocas (def. 300)")
    ap.add_argument("--patience", type=_int_range(1, 100000), default=15,
                    help="parada temprana: épocas sin mejorar el MSE held-out (def. 15)")
    ap.add_argument("--batch-size", type=_int_range(1, 65536), default=32, help="tamaño de lote (def. 32)")
    ap.add_argument("--lr", type=_float_range(0.0, 1.0, lo_open=True), default=1e-3, help="Adam (def. 1e-3)")
    ap.add_argument("--l2", type=_float_range(0.0, 1.0), default=1e-4, help="weight decay L2 (def. 1e-4)")
    ap.add_argument("--val-frac", type=_float_range(0.05, 0.5), default=0.2,
                    help="fracción held-out de las correctas (def. 0.2; por sesión: sesiones completas)")
    ap.add_argument("--split", choices=("auto", "session", "random"), default="auto",
                    help="auto (def.): por sesión con >= 3 sesiones, si no aleatorio")
    ap.add_argument("--jitter-ms", type=_float_range(0.0, 200.0), default=20.0, help="jitter máximo (def. 20 ms)")
    ap.add_argument("--noise-std", type=_float_range(0.0, 2.0), default=0.05,
                    help="ruido gaussiano en unidades z (def. 0.05)")
    ap.add_argument("--gain", type=_float_range(0.0, 0.5), default=0.1,
                    help="ganancia por canal U(1 - g, 1 + g) (def. 0.1)")
    ap.add_argument("--percentiles", type=float, nargs=3, default=list(ae.DEFAULT_PERCENTILES), metavar="P",
                    help="percentiles de T1 T2 T3 (def. 90 97 99)")
    ap.add_argument("--n-rep", type=_int_range(1, 100000), default=300,
                    help="épocas del conjunto representativo rep.npz (def. 300)")
    ap.add_argument("--syn-sessions", type=_int_range(1, 100), default=3, help="sintético: sesiones (def. 3)")
    ap.add_argument("--syn-correct", type=_int_range(1, 100000), default=200,
                    help="sintético: correctas por sesión (def. 200)")
    ap.add_argument("--syn-errors", type=_int_range(0, 100000), default=40,
                    help="sintético: con error por sesión (def. 40)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # UTF-8 también al redirigir a un pipe en Windows
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        ae.check_percentiles(args.percentiles)
    except ValueError as e:
        ap.error(str(e))
    try:
        return run(args)
    except TrainError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
