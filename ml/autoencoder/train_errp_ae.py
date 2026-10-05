#!/usr/bin/env python3
"""
Entrena el ErrP-AE del detector C2 (INTUNE) con la receta del CLAUDE.md del equipo y
deja en --out DIR lo que necesita el exportador a C (ml/c_exporter/export_to_c.py).

Datos: el dataset de C4 (ml/data/FORMAT.md, intune-c4-dataset-1.0), una sesión por .npz con
X [n, 8, 40] (µV, IIR causal, baseline [-200, 0) ms restado, media de cada 5 -> 50 Hz), y,
rejected, is_calibration, norm_mean[8] y norm_std[8]. La carga y el split los hace
errp_pipeline.py (el mismo código que training/train.py) para que no haya dos lecturas del
contrato. Para probar sin hardware: ml/data/make_synthetic.py + ml/data/build_dataset.py (C4);
si todas las sesiones son sintéticas el modelo sale como PLACEHOLDER ("placeholder": true).

Receta (FORMAT.md §6 y CLAUDE.md):
  - entrenamiento SOLO con y == 0, no rechazadas y no is_calibration; los errores (y == 1)
    solo para evaluar; las manuales (y == -1) no se usan;
  - split POR SESIÓN: test y validación son sesiones enteras (por defecto la última y la
    penúltima; --test-session / --val-session). Con < 3 sesiones el split es cronológico
    dentro de la sesión y se avisa: solo sirve para probar el código;
  - normalización z-score por canal con norm_mean/norm_std de CADA sesión (sus épocas de
    calibración, como el S3 en vivo), no por época;
  - aumento de datos en cada pasada: jitter +-20 ms = +-1 muestra a 50 Hz replicando el borde,
    ganancia por canal U(0.9, 1.1) y ruido gaussiano pequeño (desv. 0.05 en unidades z,
    solo en la entrada: el objetivo es la época sin ruido);
  - pérdida MSE, Adam + weight decay L2 en los kernels, parada temprana por el MSE de las
    correctas limpias de la sesión de VALIDACIÓN (se restauran los mejores pesos);
  - umbrales T1/T2/T3 = p90/p97/p99 del score de las épocas is_calibration (limpias) de la
    sesión de TEST, como calibra el S3; nunca de épocas de entrenamiento.

Salidas en --out DIR (se sobrescriben; cada archivo se escribe de forma atómica):
  model.keras  modelo Keras 3 con los mejores pesos (carga sin objetos personalizados:
               keras.models.load_model(ruta, compile=False))
  config.json  {"arch", "channels", "fs_hz": 250, "decim": 5, "epoch_ms": [-200, 800],
                "baseline_ms": [-200, 0], "norm": {"mean": [8], "std": [8]},
                "percentiles": [90, 97, 99], "thresholds": {"p90": T1, "p97": T2, "p99": T3},
                "score": "mse", "placeholder": bool,
                "_comentario" y "training" (informativos: el exportador los ignora)}
               norm = promedio de norm_mean/norm_std de las sesiones de entrenamiento (solo hasta
               que el S3 calibra); thresholds = calibración de la sesión de test.
               Las claves de "thresholds" salen de errp_ae.percentile_key (p99.5 -> "p99.5").
  rep.npz      "z" [n, 8, 40] float32: épocas correctas de ENTRENAMIENTO normalizadas, sin aumento
               (hasta --n-rep), conjunto representativo de la PTQ int8.
  val.npz      sesión de test, nunca usada para entrenar: sus épocas de calibración limpias
               (is_error = False, de ellas salen T1-T3) y sus errores limpios fuera de la
               calibración (is_error = True):
               "z" [M, 8, 40] float32 normalizadas (entrada del modelo), "is_error" [M] bool,
               "epochs" [M, 8, 40] float32 (uV, antes de normalizar), "mean"/"std" [M, 8] float32
               (estadísticas con que se normalizó cada época: z = normalize(epochs, mean, std)),
               "score" [M] float32 (score del modelo float) y "session" [M] int64.
  rep.npz y val.npz son deterministas byte a byte; model.keras no (Keras guarda la fecha).

Determinismo: misma máquina, semilla y datos -> mismos pesos y archivos. Antes de importar
TensorFlow el script fija el proceso a una CPU (en CPUs híbridas P/E el blocking de las GEMM de
Eigen depende del tipo de núcleo) y TF_ENABLE_ONEDNN_OPTS=0 por defecto; después, un hilo
intra/inter-op y op determinism.

Uso:
  python ml/autoencoder/train_errp_ae.py --out runs/r1                       # ml/data/processed/*.npz
  python ml/autoencoder/train_errp_ae.py --data s1.npz s2.npz s3.npz --out runs/r1 [--arch dense]
  python ml/autoencoder/train_errp_ae.py --data DIR --test-session synth_s4 --val-session synth_s3 --out runs/r1
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import errp_ae as ae  # noqa: E402
import errp_pipeline as ep  # noqa: E402  (carga y split del dataset de C4)

SCRIPT_REL = "ml/autoencoder/train_errp_ae.py"
EXPORTER_REL = "ml/c_exporter/export_to_c.py"
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


# ---- Datos (contrato de C4: ml/data/FORMAT.md) ----

@dataclass
class Data:
    sessions: List[ep.Session]          # todas las sesiones cargadas (orden de archivo)
    train: ep.Split                     # correctas limpias no-calibración de las sesiones de entrenamiento
    val: ep.Split                       # correctas limpias de la sesión de validación (parada temprana)
    test: ep.Session                    # sesión de test (umbrales con su calibración)
    test_idx: np.ndarray                # acciones evaluadas de la sesión de test (sin calibración)
    note: str                           # descripción del split
    synthetic: bool                     # todas las sesiones son sintéticas (C4: subject == "synthetic")
    source: str                         # descripción (nombres de sesión + sha256)


def _session_meta(s: ep.Session) -> Tuple[str, str]:
    """(subject, sha256 corto) de la sesión."""
    with np.load(s.path, allow_pickle=False) as d:
        subject = str(d["subject"]) if "subject" in d.files else ""
    return subject, hashlib.sha256(Path(s.path).read_bytes()).hexdigest()[:16]


def load_data(paths: Sequence[Path], test: Optional[str], val: Optional[str]) -> Data:
    try:
        sessions = ep.load_sessions(paths)
        tr, va, s_test, test_idx, note = ep.split_sessions(sessions, test, val)
    except (SystemExit, KeyError) as e:  # errp_pipeline usa SystemExit para datos inválidos
        raise TrainError(f"dataset de C4: {e}") from None
    if len(tr) < MIN_TRAIN_EPOCHS:
        raise TrainError(f"hay {len(tr)} épocas correctas de entrenamiento; hacen falta al menos {MIN_TRAIN_EPOCHS}")
    if len(va) < 1:
        raise TrainError("la sesión de validación no tiene épocas correctas limpias")
    n_cal = int(np.sum(s_test.cal_mask))
    if n_cal < ae.CALIB_MIN_SCORES:
        raise TrainError(f"la sesión de test {s_test.name} tiene {n_cal} épocas de calibración limpias; "
                         f"hacen falta >= {ae.CALIB_MIN_SCORES} para T1-T3")
    meta = [_session_meta(s) for s in sessions]
    synthetic = all(subj == "synthetic" or "synth" in s.name for s, (subj, _) in zip(sessions, meta))
    source = ", ".join(f"{s.name} (sha256 {sha})" for s, (_, sha) in zip(sessions, meta))
    return Data(sessions, tr, va, s_test, test_idx, note, synthetic, source)


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


def jitter_samples(aug: Augment) -> int:
    """Desplazamiento máximo en muestras de X (50 Hz): 20 ms -> 1 (FORMAT.md §6 regla 6)."""
    return int(round(aug.jitter_ms * (ae.FS_HZ / ae.DECIM) / 1000.0))


def augmented(sp: ep.Split, idx: np.ndarray, aug: Augment,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Una pasada de aumento: (entrada con ruido, objetivo sin ruido), ambos z [n, 8, 40].
    Jitter y ganancia en µV; normalización con las estadísticas de la sesión de cada época."""
    n = idx.size
    jmax = jitter_samples(aug)
    shifts = rng.integers(-jmax, jmax + 1, size=n) if jmax > 0 else np.zeros(n, dtype=np.int64)
    e = shift_edge(sp.X[idx], shifts)
    if aug.gain > 0:
        e = e * rng.uniform(1.0 - aug.gain, 1.0 + aug.gain, (n, ae.N_CH, 1)).astype(np.float32)
    y = ae.normalize(e, sp.mean[idx], sp.std[idx])
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


def train(data: Data, z_val: np.ndarray, args: argparse.Namespace) -> TrainResult:
    """Bucle propio: una pasada de aumento por época, lotes con train_on_batch en un orden
    barajado con la semilla, MSE de la sesión de validación y parada temprana con restauración
    de los mejores pesos."""
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
    train_idx = np.arange(len(data.train))
    print(f"Entrenando {args.arch} ({model.count_params()} parámetros): {len(data.train)} épocas correctas, "
          f"validación {len(data.val)}, lote {bs}, máx. {args.epochs} épocas, paciencia {args.patience}")
    for epoch in range(1, args.epochs + 1):
        x, y = augmented(data.train, train_idx, aug, rng)
        perm = rng.permutation(x.shape[0])
        x, y = x[perm][..., None], y[perm][..., None]
        model.reset_metrics()  # la pérdida devuelta es la media de la época (ponderada por lote)
        loss = math.nan
        for b in range(0, x.shape[0], bs):
            loss = float(np.asarray(model.train_on_batch(x[b:b + bs], y[b:b + bs])).ravel()[0])
        val = float(np.mean(predict_scores(model, z_val), dtype=np.float64))
        if not (math.isfinite(loss) and math.isfinite(val)):
            raise TrainError(f"época {epoch}: pérdida no finita (train {loss}, validación {val}); bajar --lr")
        improved = val < best_val
        if improved:
            best_val, best_epoch, best_weights = val, epoch, model.get_weights()
        if improved or epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  época {epoch:4d}  pérdida train (con L2) {loss:.5f}  MSE validación {val:.5f}"
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


def build_config(args: argparse.Namespace, data: Data, res: TrainResult, thresholds: Tuple[float, float, float],
                 norm_mean: np.ndarray, norm_std: np.ndarray, n_rep: int, n_macs: int,
                 test_metrics: Mapping[str, object]) -> Dict[str, object]:
    pcts = [float(p) for p in args.percentiles]
    train_sessions = sorted({s.name for s in data.sessions} - {data.test.name}) or [data.test.name]
    return {
        "_comentario": ("Generado por " + SCRIPT_REL + " con el dataset de C4 (ml/data/FORMAT.md). norm = promedio "
                        "de norm_mean/norm_std de las sesiones de entrenamiento (solo hasta que el S3 calibra); "
                        "thresholds = percentiles del score MSE de las épocas de calibración de la sesión de test, "
                        "como el S3. En el S3 se recalibra por operador y sesión."),
        "arch": args.arch,
        "channels": list(ae.CHANNELS),
        "fs_hz": ae.FS_HZ,
        "decim": ae.DECIM,
        "epoch_ms": list(ae.EPOCH_MS),
        "baseline_ms": list(ae.BASELINE_MS),
        "norm": {"mean": [f32(v) for v in norm_mean], "std": [f32(v) for v in norm_std]},
        "percentiles": [json_number(p) for p in pcts],
        "thresholds": {ae.percentile_key(p): f32(t) for p, t in zip(pcts, thresholds)},
        "score": "mse",
        "placeholder": bool(data.synthetic),
        "training": {
            "data": data.source,
            "dataset_format": "ml/data/FORMAT.md (intune-c4-dataset-1.0)",
            "split": data.note,
            "train_val_sessions": train_sessions,
            "test_session": data.test.name,
            "seed": args.seed,
            "n_correct_train": int(len(data.train)),
            "n_correct_val": int(len(data.val)),
            "n_test_calibration": int(np.sum(data.test.cal_mask)),
            "norm": "por sesión (norm_mean/norm_std de C4)",
            "epochs_max": args.epochs,
            "epochs_run": res.epochs_run,
            "best_epoch": res.best_epoch,
            "early_stopping": res.stopped_early,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l2": args.l2,
            "augment": {"jitter_ms": args.jitter_ms, "jitter_samples": jitter_samples(Augment(args.jitter_ms)),
                        "jitter_fs_hz": ae.FS_HZ // ae.DECIM, "noise_std": args.noise_std, "gain": args.gain},
            "val_mse": f32(res.best_val_mse),
            "n_params": int(res.model.count_params()),
            "n_macs": n_macs,
            "n_rep": n_rep,
            "test": dict(test_metrics),
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
    data = load_data(args.data, args.test_session, args.val_session)
    print(f"Datos: {len(data.sessions)} sesiones de C4; {data.note}")
    z_val = ae.normalize(data.val.X, data.val.mean, data.val.std)
    z_train = ae.normalize(data.train.X, data.train.mean, data.train.std)
    if not (np.all(np.isfinite(z_val)) and np.all(np.isfinite(z_train))):
        raise TrainError("la normalización produjo valores no finitos (¿épocas enormes?)")
    res = train(data, z_val, args)
    model = res.model

    # Sesión de test, como el S3: umbrales con SUS épocas de calibración limpias
    t = data.test
    cal_idx = np.flatnonzero(t.cal_mask)
    ev = data.test_idx[~t.rejected[data.test_idx]]              # evaluadas, limpias (y = 0 / 1)
    ev_ok, ev_err = ev[t.y[ev] == 0], ev[t.y[ev] == 1]

    def z_of(idx: np.ndarray) -> np.ndarray:
        return ae.normalize(t.X[idx], t.mean, t.std)

    s_cal = predict_scores(model, z_of(cal_idx))
    try:
        thresholds = ae.thresholds_from_scores(s_cal, args.percentiles)
    except ValueError as e:
        raise TrainError(f"umbrales con la calibración de {t.name}: {e}") from None
    s_ok = predict_scores(model, z_of(ev_ok))
    s_err = predict_scores(model, z_of(ev_err))
    test_metrics: Dict[str, object] = {"n_eval_correct": int(ev_ok.size), "n_eval_error": int(ev_err.size)}
    if ev_ok.size and ev_err.size:
        test_metrics["auc"] = f32(auc(s_err, s_ok))
        test_metrics["detection_at_1pct_fa"] = f32(np.mean(s_err > np.percentile(s_ok, 99)))

    rng_rep = np.random.default_rng([args.seed, 3])
    rep_idx = np.sort(rng_rep.choice(len(data.train), size=min(args.n_rep, len(data.train)), replace=False))
    rep = {"z": z_train[rep_idx]}
    val_idx = np.concatenate([cal_idx, ev_err])
    sess_id = [s.name for s in data.sessions].index(t.name)
    val: Dict[str, np.ndarray] = {
        "z": z_of(val_idx),
        "is_error": t.y[val_idx] == 1,
        "epochs": t.X[val_idx],
        "mean": np.repeat(t.mean[None, :], val_idx.size, axis=0),
        "std": np.repeat(t.std[None, :], val_idx.size, axis=0),
        "score": np.concatenate([s_cal, s_err]).astype(np.float32),
        "session": np.full(val_idx.size, sess_id, dtype=np.int64),
    }
    train_names = set(s.name for s in data.sessions) - {t.name} or {t.name}
    norm_mean = np.mean([s.mean for s in data.sessions if s.name in train_names], axis=0)
    norm_std = np.mean([s.std for s in data.sessions if s.name in train_names], axis=0)
    n_macs = ae.count_macs(model)
    config = build_config(args, data, res, thresholds, norm_mean, norm_std, int(rep_idx.size), n_macs, test_metrics)
    write_outputs(args.out, model, args.arch, args.l2, config, rep, val)

    # ---- Resumen ----
    pcts = [float(p) for p in args.percentiles]
    print("Entrenamiento ErrP-AE completado:")
    print(f"  arquitectura  : {args.arch} ({model.count_params()} parámetros, {n_macs} MACs)")
    print(f"  datos         : {data.source}")
    print(f"  split         : {data.note}")
    print(f"                  {len(data.train)} correctas de entrenamiento / {len(data.val)} de validación")
    print("  normalización : por sesión (norm_mean/norm_std de C4); config.json norm = promedio de las de entrenamiento")
    print(f"  épocas        : mejor {res.best_epoch} de {res.epochs_run} "
          f"({'parada temprana' if res.stopped_early else 'SIN parada temprana'}, paciencia {args.patience}), "
          f"{res.seconds:.1f} s")
    print(f"  MSE validación: {res.best_val_mse:.5f} (correctas limpias de la sesión de validación)")
    print("  umbrales      : " + "  ".join(f"{ae.percentile_key(p)}={f32(v)}" for p, v in zip(pcts, thresholds))
          + f"  (calibración de {t.name}, n = {s_cal.size})")
    print(f"  niveles 0/1/2/3 correctas test (n={ev_ok.size:4d}) : {fmt_rates(level_rates(s_ok, thresholds))}")
    if ev_err.size:
        print(f"  niveles 0/1/2/3 con error test (n={ev_err.size:4d}) : {fmt_rates(level_rates(s_err, thresholds))}")
    if "auc" in test_metrics:
        print(f"  AUC error vs correctas (test): {test_metrics['auc']:.3f}  "
              f"detección @1 % FA: {100 * test_metrics['detection_at_1pct_fa']:.1f} %")
    print(f"  placeholder   : {'SÍ (datos sintéticos: NO usar con operadores reales)' if data.synthetic else 'no'}")
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
    ap.add_argument("--data", type=Path, nargs="*", default=[],
                    help="sesiones .npz de C4 o carpetas (def. ml/data/processed)")
    ap.add_argument("--test-session", help="sesión de test (def. la última)")
    ap.add_argument("--val-session", help="sesión de validación para la parada temprana (def. la penúltima)")
    ap.add_argument("--out", type=Path, required=True, help="directorio de salida")
    ap.add_argument("--arch", choices=ae.ARCHS, default="errp_conv", help="arquitectura (def. errp_conv)")
    ap.add_argument("--seed", type=_int_range(0, 2 ** 31 - 1), default=0, help="semilla (def. 0)")
    ap.add_argument("--epochs", type=_int_range(1, 100000), default=300, help="máximo de épocas (def. 300)")
    ap.add_argument("--patience", type=_int_range(1, 100000), default=15,
                    help="parada temprana: épocas sin mejorar el MSE de validación (def. 15)")
    ap.add_argument("--batch-size", type=_int_range(1, 65536), default=32, help="tamaño de lote (def. 32)")
    ap.add_argument("--lr", type=_float_range(0.0, 1.0, lo_open=True), default=1e-3, help="Adam (def. 1e-3)")
    ap.add_argument("--l2", type=_float_range(0.0, 1.0), default=1e-4, help="weight decay L2 (def. 1e-4)")
    ap.add_argument("--jitter-ms", type=_float_range(0.0, 200.0), default=20.0, help="jitter máximo (def. 20 ms)")
    ap.add_argument("--noise-std", type=_float_range(0.0, 2.0), default=0.05,
                    help="ruido gaussiano en unidades z (def. 0.05)")
    ap.add_argument("--gain", type=_float_range(0.0, 0.5), default=0.1,
                    help="ganancia por canal U(1 - g, 1 + g) (def. 0.1)")
    ap.add_argument("--percentiles", type=float, nargs=3, default=list(ae.DEFAULT_PERCENTILES), metavar="P",
                    help="percentiles de T1 T2 T3 (def. 90 97 99)")
    ap.add_argument("--n-rep", type=_int_range(1, 100000), default=300,
                    help="épocas del conjunto representativo rep.npz (def. 300)")
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
