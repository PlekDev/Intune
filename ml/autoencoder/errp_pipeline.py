"""Datos del Detector (C2) a partir del dataset de C4 (ml/data/FORMAT.md).

C4 entrega una sesión por .npz con X [n, 8, 40] en µV: IIR causal (C1), ventana
[-200, +800) ms, baseline [-200, 0) ms restado y media de cada 5 -> 50 Hz. Aquí
NO se vuelve a preprocesar: solo se normaliza por canal con norm_mean/norm_std de
CADA sesión (sus épocas de calibración, como el S3 en vivo) y se arman los splits.

Reglas de FORMAT.md §6:
  train:  y == 0, no rechazadas, no is_calibration
  umbrales: épocas is_calibration (limpias) de la sesión evaluada
  split por sesión; early stopping con correctas limpias de sesiones de validación.

preprocess_window() es la referencia de los pasos 3-4 de FORMAT.md para la prueba
del DSP del S3 (firmware/detector_s3/test/host_dsp_test.c).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 250
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
N_CH = len(CHANNELS)
PRE = 50          # [-200, 0) ms a 250 Hz
POST = 200        # [0, 800) ms a 250 Hz
DECIM = 5
N_T = POST // DECIM  # 40
STD_FLOOR = 1e-3  # µV, como build_dataset.py

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@dataclass
class Session:
    name: str
    path: Path
    X: np.ndarray           # [n, 8, 40] µV
    y: np.ndarray           # 0 correcta, 1 error, -1 manual
    rejected: np.ndarray
    is_cal: np.ndarray
    mean: np.ndarray        # [8]
    std: np.ndarray         # [8]
    evt_counter: np.ndarray
    norm_valid: bool

    @property
    def train_mask(self):
        return (self.y == 0) & ~self.rejected & ~self.is_cal

    @property
    def cal_mask(self):
        return self.is_cal & ~self.rejected

    @property
    def eval_mask(self):
        """Acciones que se evalúan (incluye rechazadas para la máquina de estados)."""
        return (self.y >= 0) & ~self.is_cal

    def z(self, idx=None) -> np.ndarray:
        x = self.X if idx is None else self.X[idx]
        return normalize(x, self.mean, self.std)


def load_session(path: Path) -> Session:
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)
    if X.ndim != 3 or X.shape[1:] != (N_CH, N_T):
        raise SystemExit(f"{path}: X debe ser [n, {N_CH}, {N_T}], es {X.shape}")
    if list(map(str, d["ch_names"])) != CHANNELS:
        raise SystemExit(f"{path}: canales {list(d['ch_names'])} != {CHANNELS}")
    std = np.maximum(d["norm_std"].astype(np.float32), STD_FLOOR)
    s = Session(name=str(d["session"]), path=path, X=X, y=d["y"].astype(np.int8),
                rejected=d["rejected"].astype(bool), is_cal=d["is_calibration"].astype(bool),
                mean=d["norm_mean"].astype(np.float32), std=std,
                evt_counter=d["evt_counter"].astype(np.int64), norm_valid=bool(d["norm_valid"]))
    if not s.norm_valid:
        print(f"AVISO {s.name}: norm_valid = False (< 20 épocas de calibración)", file=sys.stderr)
    return s


def load_sessions(paths) -> list[Session]:
    files = []
    for p in map(Path, paths or [DEFAULT_DATA_DIR]):
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no hay sesiones .npz en {paths or DEFAULT_DATA_DIR}")
    return [load_session(f) for f in files]


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score por canal (no por época): [n, 8, 40] con mean/std [8] o [n, 8]."""
    m = mean[..., :, None] if mean.ndim == 2 else mean[None, :, None]
    s = std[..., :, None] if std.ndim == 2 else std[None, :, None]
    return ((x - m) / s).astype(np.float32)


@dataclass
class Split:
    """Entrenamiento y validación: épocas correctas limpias (µV + stats de su sesión)."""
    X: np.ndarray
    mean: np.ndarray  # [n, 8]
    std: np.ndarray

    @property
    def z(self):
        return normalize(self.X, self.mean, self.std)

    def __len__(self):
        return len(self.X)


def _split_of(parts) -> Split:
    xs = [s.X[i] for s, i in parts]
    ms = [np.repeat(s.mean[None], len(i), 0) for s, i in parts]
    ss = [np.repeat(s.std[None], len(i), 0) for s, i in parts]
    return Split(np.concatenate(xs), np.concatenate(ms), np.concatenate(ss))


def split_sessions(sessions: list[Session], test: str | None = None, val: str | None = None):
    """Devuelve (train: Split, val: Split, test: Session, test_idx, nota).

    >= 3 sesiones: test y val son sesiones enteras (por defecto las dos últimas).
    1-2 sesiones: no hay split por sesión posible; se parte en orden cronológico
    y se avisa (solo sirve para probar el código, FORMAT.md §6 regla 3).
    test_idx = acciones de la sesión de test que se evalúan (sin calibración).
    """
    by_name = {s.name: s for s in sessions}
    if len(sessions) >= 3:
        s_test = by_name[test] if test else sessions[-1]
        rest = [s for s in sessions if s is not s_test]
        s_val = by_name[val] if val else rest[-1]
        train = [s for s in rest if s is not s_val]
        return (_split_of([(s, np.flatnonzero(s.train_mask)) for s in train]),
                _split_of([(s_val, np.flatnonzero(s_val.train_mask))]),
                s_test, np.flatnonzero(s_test.eval_mask),
                f"por sesión: train {[s.name for s in train]}, val {s_val.name}, test {s_test.name}")

    note = "AVISO: < 3 sesiones, split cronológico dentro de la sesión (solo para probar el código)"
    print(note, file=sys.stderr)
    if len(sessions) == 2:
        s_tr, s_test = sessions
        idx = np.flatnonzero(s_tr.train_mask)
        cut = int(len(idx) * 0.8)
        return (_split_of([(s_tr, idx[:cut])]), _split_of([(s_tr, idx[cut:])]),
                s_test, np.flatnonzero(s_test.eval_mask), note)
    s = sessions[0]
    ev = np.flatnonzero(s.eval_mask)          # cronológico, sin calibración
    a, b = int(len(ev) * 0.70), int(len(ev) * 0.85)
    tr, va, te = ev[:a], ev[a:b], ev[b:]
    keep = s.train_mask
    return (_split_of([(s, tr[keep[tr]])]), _split_of([(s, va[keep[va]])]), s, te, note)


def augment(sp: Split, idx: np.ndarray, rng: np.random.Generator, noise_std: float = 0.1) -> np.ndarray:
    """CLAUDE.md: jitter ±20 ms (= ±1 muestra a 50 Hz, replicando el borde), ganancia
    por canal ±10 % (en µV, antes de normalizar) y ruido gaussiano (unidades z)."""
    x = sp.X[idx] * rng.uniform(0.9, 1.1, size=(len(idx), N_CH, 1)).astype(np.float32)
    shift = rng.integers(-1, 2, size=len(idx))
    out = x.copy()
    m = shift == 1    # retrasar: x[t-1], borde izquierdo replicado
    out[m, :, 1:] = x[m, :, :-1]
    m = shift == -1   # adelantar: x[t+1], borde derecho replicado
    out[m, :, :-1] = x[m, :, 1:]
    z = normalize(out, sp.mean[idx], sp.std[idx])
    return z + rng.normal(0.0, noise_std, size=z.shape).astype(np.float32)


def to_model(z: np.ndarray) -> np.ndarray:
    """[n, 8, 40] -> [n, 8, 40, 1] (NHWC: alto = electrodos, ancho = tiempo)."""
    return z[..., None]


def mse_score(z: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """MSE por época sobre todos los ejes salvo el batch."""
    return ((recon.reshape(z.shape) - z) ** 2).reshape(len(z), -1).mean(axis=1)


def preprocess_window(w: np.ndarray) -> np.ndarray:
    """Referencia de FORMAT.md §2 pasos 3-4: ventana [8, 250] µV filtrados
    ([-200, +800) ms a 250 Hz) -> [8, 40] (baseline restado, media de cada 5)."""
    w = np.asarray(w, dtype=np.float32)
    base = w[:, :PRE].mean(axis=1, keepdims=True)
    return (w[:, PRE:PRE + POST] - base).reshape(N_CH, N_T, DECIM).mean(axis=-1)
