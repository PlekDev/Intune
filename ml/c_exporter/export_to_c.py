#!/usr/bin/env python3
"""
Exporta el ErrP-AE del detector C2 (INTUNE) a los headers C del ESP32-S3: modelo Keras ->
TensorFlow Lite int8 (TFLite Micro en el S3) + metadatos + vectores dorados. CONTRACT v3, sección 6.

Entradas (las escribe ml/autoencoder/train_errp_ae.py en su --out DIR):
  model.keras  modelo Keras 3 de ml/autoencoder/errp_ae.py (build_errp_conv_ae o build_dense_ae),
               entrada y salida (8, 40, 1) = [electrodo][tiempo]
  config.json  {"arch": "errp_conv" | "dense", "channels": [Fz, C3, Cz, C4, Pz, PO7, Oz, PO8],
                "fs_hz": 250, "decim": 5, "epoch_ms": [-200, 800], "baseline_ms": [-200, 0],
                "norm": {"mean": [8], "std": [8]}, "percentiles": [90, 97, 99],
                "thresholds": {"p90": T1, "p97": T2, "p99": T3}, "score": "mse",
                "placeholder": bool}           (todas obligatorias; las demás claves se ignoran)
  rep.npz      "z" [n, 8, 40] float32: épocas CORRECTAS normalizadas (conjunto representativo
               de la cuantización post-entrenamiento int8; unos cientos)
  val.npz      "z" [M, 8, 40] float32 + "is_error" [M] bool: correctas held-out y con error, no
               usadas para entrenar ("score" [M] opcional: solo para avisar de incoherencias)

Pasos (si algo falla: mensaje en español, exit 1 y NO se escribe nada):
  1. valida config.json, rep.npz, val.npz y el modelo;
  2. convierte con tf.lite.TFLiteConverter: PTQ int8 completa (TFLITE_BUILTINS_INT8), conjunto
     representativo = rep.npz como [1, 8, 40, 1] float32, entrada y salida int8 y lote fijo 1
     (errp_ae.inference_model);
  3. comprueba el .tflite: un subgrafo; solo ops CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D,
     FULLY_CONNECTED, RESHAPE y RELU (las que registra el runner TFLM del S3); tensores int8 o
     int32; entrada y salida int8 [1, 8, 40, 1] con cuantización por tensor; y que el modelo es
     la arquitectura "arch" de errp_ae (mismas capas, formas y activaciones);
  4. score float (Keras) frente a score int8 en val.npz: correlación de Pearson >= --min-corr
     (CLAUDE.md: 0.98), |score_f - score_q| máximo y acuerdo de nivel con los umbrales por
     defecto. El score int8 sigue la cadena exacta del motor C (errp_ae: quantize -> modelo ->
     dequantize -> score) con el runtime de --golden-runtime (abajo);
  5. genera los vectores dorados con ese runtime y escribe las salidas de forma atómica
     (temporales en el mismo directorio + os.replace, solo al final).

Runtime de q_out (--golden-runtime):
  tflm         (por defecto) emulación entera, bit a bit, de los kernels de referencia int8 de
               TFLite Micro (esp-tflite-micro 1.4.1); verificada contra TFLM real en el PC.
  builtin_ref  tf.lite.Interpreter con kernels de REFERENCIA (OpResolverType.BUILTIN_REF), lo
               que pedía el CONTRACT v3.
  En TF 2.21, el FULLY_CONNECTED de BUILTIN_REF redondea los empates de la recuantización de
  otra forma que TFLM: ~1-2 % de los valores de q_out difieren en 1 LSB y el nivel B de
  test_c_engine.c (TFLM real) los ve. Por eso los goldens salen por defecto de la emulación; el
  exportador contrasta siempre los dos runtimes y lo deja escrito en los headers.

Salidas (generadas y versionadas en el repo: NO editarlas a mano):
  firmware/detector_s3/include/autoencoder_weights.h    .tflite como array C + metadatos
                                                       (solo lo incluye autoencoder_tflm.cc)
  firmware/detector_s3/test/autoencoder_test_vectors.h  vectores dorados de test_c_engine.c
  --tflite-out F: además el .tflite (los mismos bytes que AE_MODEL_TFLITE)
Los dos headers llevan el mismo AE_MODEL_ID (16 hex del sha256 del .tflite + calibración por
defecto + percentiles): el test en C falla si quedan desfasados.

AE_TENSOR_ARENA_BYTES: --arena-bytes N con la arena MEDIDA en un build TFLM (el nivel B de
test_c_engine.c imprime la arena usada) más margen. Sin --arena-bytes se escribe la estimación
DEFAULT_ARENA_BYTES (ver su comentario: medida en un build TFLM de PC + margen).

Determinismo: mismas entradas + misma máquina y versiones -> mismos bytes (proceso fijado a una
CPU antes de importar TensorFlow, un hilo, TF_ENABLE_ONEDNN_OPTS=0, op determinism, semillas).
Los headers no llevan fechas ni rutas absolutas (solo nombres base y sha256 de las entradas).

Uso:
  python ml/c_exporter/export_to_c.py --model DIR/model.keras --config DIR/config.json \\
         --rep DIR/rep.npz --val DIR/val.npz
  python ml/c_exporter/export_to_c.py ... --out /tmp/w.h --test-vectors /tmp/tv.h --tflite-out /tmp/m.tflite
  python ml/c_exporter/export_to_c.py ... --arena-bytes 12288 [--golden-runtime builtin_ref]
  python ml/c_exporter/export_to_c.py --make-fixture /tmp/fx [--arch dense] [--seed 0]   # pesos ALEATORIOS
  python ml/c_exporter/export_to_c.py --selftest                                         # no toca el repo
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import warnings
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AE_DIR = str(Path(__file__).resolve().parents[1] / "autoencoder")
if _AE_DIR not in sys.path:
    sys.path.insert(0, _AE_DIR)

import errp_ae as ae  # noqa: E402  (numpy solo; TensorFlow se importa al convertir)

# ---- Rutas y constantes ----

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "firmware" / "detector_s3" / "include" / "autoencoder_weights.h"
DEFAULT_TEST_VECTORS = REPO_ROOT / "firmware" / "detector_s3" / "test" / "autoencoder_test_vectors.h"
SCRIPT_REL = "ml/c_exporter/export_to_c.py"

DEFAULT_MIN_CORR = 0.98       # CLAUDE.md: correlación score float vs int8 >= 0.98
# Estimación por defecto de AE_TENSOR_ARENA_BYTES (sin --arena-bytes). Arena usada por
# MicroInterpreter (arena_used_bytes) medida en un build TFLM de PC (esp-tflite-micro 1.4.1,
# x86-64, kernels de referencia; nivel B de test_c_engine.c) con los modelos de este exportador
# (fixtures y modelos entrenados): errp_conv 9184 B, dense 6016 B. 16 KB deja ~7 KB de margen
# para los buffers de trabajo de los kernels ESP-NN del S3 y las alineaciones (en el S3, de 32
# bits, las estructuras de TFLM ocupan menos). El valor bueno es el MEDIDO en el S3 (nivel C) +
# margen, con --arena-bytes.
DEFAULT_ARENA_BYTES = 16384
ARENA_NOTE = "arena medida en PC x86-64, TFLM 1.4.1, kernels de referencia: errp_conv 9184 B, dense 6016 B"
ARENA_MIN, ARENA_MAX = 1024, 16 * 1024 * 1024
MIN_REP = 20                  # épocas mínimas del conjunto representativo
WARN_REP = 100                # aviso por debajo (la PTQ calibra rangos con pocas épocas)
MIN_VAL = 20                  # épocas mínimas de val.npz (correlación)
NEAR_REL = 1e-4               # |score - Tk| <= 1e-4 * Tk -> AE_TV_NEAR_THRESHOLD = 1
CHECK_REL = 1e-3              # val.npz "score" y umbrales frente a lo recalculado (aviso)
PREDICT_BATCH = 256

TV_N_WIN = 3                  # ventanas crudas (goldens de preprocesado)
TV_N_VAL_CORRECT = 3          # épocas correctas de val.npz en los goldens
TV_N_VAL_ERROR = 3            # épocas con error de val.npz en los goldens
TV_PER_BAND = 2               # vectores dirigidos por banda de nivel no vacía
TV_CAL_N_EPOCHS = 6           # épocas de los goldens de mean/std (Welford)
TV_CAL_N_SCORES = 60          # scores de los goldens de umbrales (>= AE_CALIB_MIN_SCORES)
DIRECTED_TRIES = 24           # direcciones aleatorias por vector dirigido
BISECT_ITERS = 60
SATURATION_Z = 1000.0         # vector de saturación: z = 1000 * d -> q_in en -128 y 127
# q_out de los goldens: "tflm" (por defecto: emulación entera de los kernels de referencia de TFLite
# Micro, bit a bit con TFLM real) o "builtin_ref" (lo que pedía el CONTRACT v3: tf.lite.Interpreter con
# OpResolverType.BUILTIN_REF, que en TF 2.21 redondea distinto los empates de FULLY_CONNECTED y hace
# fallar el nivel B de test_c_engine.c por 1 LSB: ver TflmReference)
DEFAULT_GOLDEN_RUNTIME = "tflm"
# Contraste BUILTIN_REF / emulación en val.npz: los empates de FULLY_CONNECTED cambian ~1-2 % de los
# valores en 1 LSB; por encima de esto la emulación (o el modelo) no es lo que se cree -> error en
# --golden-runtime tflm, aviso en builtin_ref
XCHECK_MAX_LSB = 16
XCHECK_MAX_FRAC = 0.25        # fracción máxima de valores de q_out distintos

VALUES_PER_LINE = 6           # floats por línea (líneas <= 120 caracteres incluso en arrays 3-D)
INTS_PER_LINE = 16
BYTES_PER_LINE = 16

FIXTURE_FILES = ("model.keras", "config.json", "rep.npz", "val.npz")
FIXTURE_N_TRAIN = 300         # épocas correctas "de entrenamiento" (rep.npz y norm por defecto)
FIXTURE_N_VAL_CORRECT = 200
FIXTURE_N_VAL_ERROR = 40
FIXTURE_DATE = "1980-01-01@00:00:00"  # fecha fija de metadata.json en el model.keras del fixture

LEVEL_NAMES = ("Mild", "Moderate", "Severe")
ALLOWED_OPS = tuple(ae.ALLOWED_TFLITE_OPS)
ALLOWED_TENSOR_TYPES = ("INT8", "INT32")
IO_SHAPE = (1,) + ae.INPUT_SHAPE  # (1, 8, 40, 1)
CONFIG_KEYS = ("arch", "channels", "fs_hz", "decim", "epoch_ms", "baseline_ms", "norm", "percentiles",
               "thresholds", "score", "placeholder")


class ExportError(Exception):
    """Problema esperado (entrada inválida, verificación fallida...): mensaje claro y exit 1."""


_warning_sink: Optional[List[str]] = None


def warn(msg: str) -> None:
    """Aviso en stderr (o capturado durante el selftest)."""
    if _warning_sink is not None:
        _warning_sink.append(msg)
    else:
        print(f"AVISO: {msg}", file=sys.stderr)


@contextlib.contextmanager
def captured_warnings():
    """Captura los avisos en una lista en vez de imprimirlos (selftest)."""
    global _warning_sink
    previous = _warning_sink
    _warning_sink = []
    try:
        yield _warning_sink
    finally:
        _warning_sink = previous


# ---- Utilidades de float32, texto y archivos ----

def to_f32(value: float) -> np.float32:
    """Redondea a float32 (fuera de rango -> inf, sin RuntimeWarning)."""
    with np.errstate(over="ignore"):
        return np.float32(value)


def c_float(value: float) -> str:
    """Literal C float32 exacto: redondeo a float32, '%.9g' (ida y vuelta exacta) y sufijo f."""
    v = to_f32(value)
    if not np.isfinite(v):
        raise ExportError(f"valor no finito ({value!r}): no se puede escribir como literal C")
    text = "%.9g" % float(v)
    if "." not in text and "e" not in text:
        text += ".0"  # '1f' no es C válido: '1.0f' ('-0' -> '-0.0f')
    return text + "f"


def c_int(value: int) -> str:
    """Entero para un #define: los negativos entre paréntesis (x - AE_IN_ZERO_POINT)."""
    v = int(value)
    return f"({v})" if v < 0 else str(v)


def short_float(value: float) -> str:
    """Texto más corto que vuelve al mismo float32 (comentarios, mensajes y JSON)."""
    return str(to_f32(value))


def json_f32(value: float) -> float:
    """Float de Python con el texto más corto que vuelve al mismo float32 (config.json)."""
    return float(short_float(value))


_COMMENT_UNSAFE = re.compile(r"[^A-Za-z0-9 _\-.,:;()\[\]{}+=<>%#&@!|^~/'\"]")


def ascii_comment(text: str, max_len: int = 64) -> str:
    """Texto seguro dentro de un comentario // de C: ASCII, sin '*' (nada de '/*' ni '*/'),
    '\\' (empalme de línea), '?' (trigrafos) ni saltos de línea. Las tildes se pliegan (ó -> o)."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    safe = _COMMENT_UNSAFE.sub("_", plain).strip()
    return safe[:max_len] or "_"


def _brief(message: str, limit: int = 300) -> str:
    """Primera línea no vacía de un mensaje de error largo (sin códigos ANSI)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(message))
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line[:limit]


def _tail(text: str, n: int = 6, limit: int = 600) -> str:
    """Últimas líneas no vacías de una salida capturada (para mensajes de error)."""
    lines = [ln.rstrip() for ln in str(text).splitlines() if ln.strip()]
    out = " | ".join(lines[-n:])
    return out[-limit:]


def _keys(d: Mapping) -> str:
    return ", ".join(repr(k) for k in list(d.keys())[:30]) or "(ninguna)"


def _short_repr(value: object, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _same_file(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_files_atomic(items: Sequence[Tuple[Path, bytes]]) -> None:
    """Escribe varios archivos casi como una transacción: primero TODOS los temporales (en el
    directorio de cada destino, con fsync) y después los os.replace. Si falla la preparación no
    se toca ningún destino y se borran los temporales."""
    pending: List[Tuple[str, Path]] = []
    try:
        for path, data in items:
            path = Path(path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
            except OSError as e:
                raise ExportError(f"no se puede escribir en {path.parent}: {e}") from None
            pending.append((tmp, path))
            with os.fdopen(fd, "wb") as f:
                f.write(bytes(data))
                f.flush()
                os.fsync(f.fileno())
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)  # mkstemp crea con 0600
        while pending:
            tmp, path = pending[0]
            os.replace(tmp, str(path))
            pending.pop(0)
    except BaseException as e:
        for tmp, _ in pending:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if isinstance(e, OSError):
            raise ExportError(f"no se pudo escribir la salida: {e}") from None
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


@contextlib.contextmanager
def captured_output(sink: Optional[List[str]] = None):
    """Redirige los descriptores 1 y 2 del proceso (también lo que escriben las bibliotecas C++
    de TensorFlow: 'Saved artifact at...', 'fully_quantize: ...') a un temporal anónimo. Al
    salir (también con excepción) añade a sink el texto capturado."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    try:
        saved = (os.dup(1), os.dup(2))
    except OSError:  # sin descriptores (p.ej. pythonw): no se captura nada
        yield
        return
    tmp = tempfile.TemporaryFile()
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        if sink is not None:
            tmp.seek(0)
            sink.append(tmp.read().decode("utf-8", "replace"))
        tmp.close()


# ---- TensorFlow (solo aquí; errp_ae y la validación de entradas no lo necesitan) ----

_TF_CACHE: Optional[tuple] = None


def import_tf():
    """TensorFlow + Keras deterministas: proceso fijado a una CPU ANTES de importar TensorFlow
    (errp_ae.pin_single_cpu: en CPUs híbridas P/E el blocking de las GEMM de Eigen depende del
    tipo de núcleo), TF_ENABLE_ONEDNN_OPTS=0 (oneDNN cambia los floats entre ejecuciones), un
    hilo intra/inter-op y op determinism. Devuelve (tf, keras)."""
    global _TF_CACHE
    if _TF_CACHE is not None:
        return _TF_CACHE
    if "tensorflow" not in sys.modules and not ae.pin_single_cpu():
        warn("no se pudo fijar el proceso a una CPU: en CPUs híbridas la conversión puede no ser "
             "determinista en los últimos bits")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        with captured_output():  # avisos de TensorFlow al importar (GPU en Windows...)
            keras = ae.keras_module()
            import tensorflow as tf
    except ImportError:
        raise ExportError("falta TensorFlow/Keras en este entorno de Python (pip install tensorflow-cpu): "
                          "el exportador lo necesita para convertir el modelo a TFLite int8") from None
    except RuntimeError as e:  # backend de Keras distinto de tensorflow
        raise ExportError(str(e)) from None
    except Exception as e:  # p.ej. DLL de TensorFlow que no carga
        raise ExportError(f"no se pudo importar TensorFlow: {type(e).__name__}: {_brief(str(e))}") from None
    if os.environ.get("TF_ENABLE_ONEDNN_OPTS") != "0":
        warn("TF_ENABLE_ONEDNN_OPTS no es 0: oneDNN puede cambiar los resultados entre ejecuciones")
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:  # el runtime de TensorFlow ya estaba inicializado en este proceso
        warn("no se pudo fijar un solo hilo en TensorFlow: la exportación puede no ser determinista")
    tf.config.experimental.enable_op_determinism()
    _TF_CACHE = (tf, keras)
    return _TF_CACHE


# ---- config.json ----

@dataclass
class ExportConfig:
    arch: str
    mean: np.ndarray                            # float32 [8]: calibración por defecto
    std: np.ndarray                             # float32 [8]
    percentiles: Tuple[float, float, float]     # p de T1, T2, T3 (90, 97, 99)
    thresholds: Tuple[float, float, float]      # floats exactamente representables en float32
    placeholder: bool


def _number(value: object) -> Optional[float]:
    """Número JSON -> float (inf si no cabe); None si no es un número (bool no cuenta)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _f32_vector(values: object, n: int, what: str) -> np.ndarray:
    if not isinstance(values, list):
        raise ExportError(f"{what} debe ser una lista de {n} números (es {type(values).__name__})")
    if len(values) != n:
        raise ExportError(f"{what} tiene {len(values)} valores; se esperaban {n}")
    out = np.empty(n, np.float32)
    for i, v in enumerate(values):
        f = _number(v)
        if f is None:
            raise ExportError(f"{what}[{i}] no es un número: {_short_repr(v)}")
        f32 = to_f32(f)
        if not np.isfinite(f32):
            raise ExportError(f"{what}[{i}] = {_short_repr(v)} no es finito en float32")
        out[i] = f32
    return out


def _same_numbers(value: object, expected: Sequence[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    nums = [_number(v) for v in value]
    return all(x is not None and x == e for x, e in zip(nums, expected))


def parse_config(raw: object, where: str) -> ExportConfig:
    """Valida config.json (formato del CONTRACT v3, sección 5). Claves extra: se ignoran."""
    if not isinstance(raw, Mapping):
        raise ExportError(f"{where}: debe ser un objeto JSON {{...}} (es {type(raw).__name__})")
    missing = [k for k in CONFIG_KEYS if k not in raw]
    if missing:
        raise ExportError(f"{where}: faltan claves {missing} (formato: python {SCRIPT_REL} --help). "
                          f"Claves encontradas: {_keys(raw)}")
    arch = raw["arch"]
    if arch not in ae.ARCHS:
        raise ExportError(f"{where}: 'arch' = {_short_repr(arch)} no válida (opciones: {', '.join(ae.ARCHS)})")
    if raw["channels"] != list(ae.CHANNELS):
        raise ExportError(f"{where}: 'channels' = {_short_repr(raw['channels'], 80)}; el motor del S3 espera "
                          f"{ae.CHANNELS} en ese orden")
    for key, expected in (("fs_hz", ae.FS_HZ), ("decim", ae.DECIM)):
        value = _number(raw[key])
        if value is None or value != expected:
            raise ExportError(f"{where}: '{key}' = {_short_repr(raw[key])}; el motor del S3 usa {expected}")
    for key, expected in (("epoch_ms", ae.EPOCH_MS), ("baseline_ms", ae.BASELINE_MS)):
        if not _same_numbers(raw[key], expected):
            raise ExportError(f"{where}: '{key}' = {_short_repr(raw[key])}; el motor del S3 usa {list(expected)}")
    norm = raw["norm"]
    if not isinstance(norm, Mapping) or "mean" not in norm or "std" not in norm:
        raise ExportError(f'{where}: \'norm\' debe ser {{"mean": [8 números], "std": [8 números]}}')
    mean = _f32_vector(norm["mean"], ae.N_CH, f"{where}: 'norm.mean'")
    std = _f32_vector(norm["std"], ae.N_CH, f"{where}: 'norm.std'")
    bad = [ae.CHANNELS[c] for c in range(ae.N_CH) if not std[c] > 0]
    if bad:
        raise ExportError(f"{where}: 'norm.std' debe ser > 0 en todos los canales (falla en {bad})")
    pct_raw = raw["percentiles"]
    if not isinstance(pct_raw, list) or any(_number(v) is None for v in pct_raw):
        raise ExportError(f"{where}: 'percentiles' debe ser una lista de 3 números, p.ej. [90, 97, 99]")
    try:
        pcts = ae.check_percentiles([_number(v) for v in pct_raw])
    except ValueError as e:
        raise ExportError(f"{where}: {e}") from None
    thr_raw = raw["thresholds"]
    if not isinstance(thr_raw, Mapping):
        raise ExportError(f"{where}: 'thresholds' debe ser un objeto {{\"p90\": T1, \"p97\": T2, \"p99\": T3}}")
    thresholds: List[float] = []
    for p in pcts:
        key = ae.percentile_key(p)
        if key not in thr_raw:
            raise ExportError(f"{where}: falta el umbral '{key}' en 'thresholds' (percentiles {list(pcts)}; "
                              f"claves encontradas: {_keys(thr_raw)})")
        f = _number(thr_raw[key])
        t = to_f32(f) if f is not None else None
        if t is None or not (np.isfinite(t) and t > 0):
            raise ExportError(f"{where}: el umbral '{key}' = {_short_repr(thr_raw[key])} debe ser un número "
                              "finito y > 0 (en float32)")
        thresholds.append(float(t))
    t1, t2, t3 = thresholds
    if not t1 <= t2 <= t3:
        shown = ", ".join(f"{ae.percentile_key(p)}={short_float(t)}" for p, t in zip(pcts, thresholds))
        raise ExportError(f"{where}: umbrales no monótonos: {shown} (se exige T1 <= T2 <= T3)")
    score = raw["score"]
    if not (isinstance(score, str) and score.strip().lower() == "mse"):
        raise ExportError(f"{where}: 'score' = {_short_repr(score)} no soportado: el motor C calcula 'mse' "
                          "(media de (z - z_hat)^2 sobre las 320 muestras normalizadas)")
    placeholder = raw["placeholder"]
    if not isinstance(placeholder, bool):
        raise ExportError(f"{where}: 'placeholder' debe ser true o false (es {_short_repr(placeholder)})")
    return ExportConfig(str(arch), mean, std, pcts, (t1, t2, t3), placeholder)


def load_config(path: Path) -> Tuple[ExportConfig, bytes]:
    path = Path(path)
    if not path.is_file():
        raise ExportError(f"no existe el config: {path}")
    data = path.read_bytes()
    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:  # json.JSONDecodeError es un ValueError
        raise ExportError(f"{path.name} no es un JSON válido: {e}") from None
    return parse_config(raw, path.name), data


# ---- rep.npz / val.npz ----

@dataclass
class ValSet:
    z: np.ndarray                               # float32 [M, 8, 40]
    is_error: np.ndarray                        # bool [M]
    score: Optional[np.ndarray]                 # float32 [M] (score float del entrenamiento) o None


def load_npz(path: Path, what: str) -> Tuple[Dict[str, np.ndarray], bytes]:
    path = Path(path)
    if not path.is_file():
        raise ExportError(f"no existe {what}: {path}")
    data = path.read_bytes()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ExportError(f"{path.name} no es un .npz (np.savez): se esperaba {what}")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {k: np.asarray(npz[k]) for k in npz.files}
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as e:
        raise ExportError(f"no se pudo leer {path.name} como .npz: {_brief(str(e))}") from None
    return arrays, data


def _epochs_z(arrays: Mapping[str, np.ndarray], key: str, where: str, min_n: int) -> np.ndarray:
    """Épocas normalizadas [N, 8, 40] (también [N, 8, 40, 1]) float32, finitas, N >= min_n."""
    if key not in arrays:
        raise ExportError(f"{where}: falta '{key}' (claves: {_keys(arrays)})")
    a = arrays[key]
    if a.ndim == 4 and a.shape[1:] == ae.INPUT_SHAPE:
        a = a[..., 0]
    if a.ndim != 3 or a.shape[1:] != (ae.N_CH, ae.N_T):
        raise ExportError(f"{where}: '{key}' con forma {a.shape}; se esperaba [N, {ae.N_CH}, {ae.N_T}]")
    if not (np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer)):
        raise ExportError(f"{where}: '{key}' no es numérico ({a.dtype})")
    if a.shape[0] < min_n:
        raise ExportError(f"{where}: '{key}' tiene {a.shape[0]} épocas; hacen falta al menos {min_n}")
    with np.errstate(over="ignore"):
        z = np.ascontiguousarray(a, dtype=np.float32)
    bad = np.flatnonzero(~np.all(np.isfinite(z), axis=(1, 2)))
    if bad.size:
        raise ExportError(f"{where}: '{key}' tiene {bad.size} épocas con valores no finitos (la primera: {bad[0]})")
    return z


def parse_rep(arrays: Mapping[str, np.ndarray], where: str) -> np.ndarray:
    z = _epochs_z(arrays, "z", where, MIN_REP)
    if z.shape[0] < WARN_REP:
        warn(f"{where}: solo {z.shape[0]} épocas en el conjunto representativo; la cuantización int8 "
             f"calibra los rangos con ellas (recomendado >= {WARN_REP})")
    return z


def parse_val(arrays: Mapping[str, np.ndarray], where: str) -> ValSet:
    z = _epochs_z(arrays, "z", where, MIN_VAL)
    n = z.shape[0]
    if "is_error" not in arrays:
        raise ExportError(f"{where}: falta 'is_error' [M] bool (claves: {_keys(arrays)})")
    is_error = arrays["is_error"]
    if is_error.shape != (n,):
        raise ExportError(f"{where}: 'is_error' con forma {is_error.shape}; se esperaba ({n},)")
    if is_error.dtype != np.bool_:
        if not np.issubdtype(is_error.dtype, np.integer) or not np.all((is_error == 0) | (is_error == 1)):
            raise ExportError(f"{where}: 'is_error' debe ser bool (o enteros 0/1)")
        is_error = is_error.astype(bool)
    score = None
    if "score" in arrays:
        s = arrays["score"]
        if s.shape == (n,) and np.issubdtype(s.dtype, np.floating):
            score = s.astype(np.float32)
        else:
            warn(f"{where}: 'score' con forma {s.shape} / tipo {s.dtype}: se ignora")
    return ValSet(z, np.ascontiguousarray(is_error, dtype=bool), score)


# ---- Modelo Keras ----

_ARCH_ATTRS = ("filters", "kernel_size", "strides", "padding", "dilation_rate", "depth_multiplier", "use_bias",
               "units", "activation", "pool_size", "target_shape", "data_format", "center", "scale", "max_value",
               "negative_slope", "threshold")


def load_keras_model(path: Path, keras):
    path = Path(path)
    if not path.is_file():
        raise ExportError(f"no existe el modelo: {path}")
    try:
        with captured_output():
            model = keras.models.load_model(str(path), compile=False)
    except Exception as e:
        raise ExportError(f"no se pudo cargar {path.name} con keras.models.load_model (se esperaba un "
                          f".keras de errp_ae): {type(e).__name__}: {_brief(str(e))}") from None
    if not hasattr(model, "input_shape") or not hasattr(model, "output_shape"):
        raise ExportError(f"{path.name} no es un keras.Model")
    shapes = (model.input_shape, model.output_shape)
    for kind, shape in zip(("entrada", "salida"), shapes):
        if (not isinstance(shape, tuple) or len(shape) != 4 or tuple(shape[1:]) != ae.INPUT_SHAPE
                or shape[0] not in (None, 1)):
            raise ExportError(f"{path.name}: {kind} con forma {shape}; se esperaba (None, 8, 40, 1) "
                              "([electrodo][tiempo], NHWC)")
    return model


def _layer_signature(model) -> List[Tuple[str, tuple, tuple]]:
    """(clase, atributos relevantes de get_config, formas de los pesos) de cada capa."""
    sig = []
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "InputLayer":
            continue
        cfg = layer.get_config()
        attrs = []
        for key in _ARCH_ATTRS:
            if key in cfg:
                value = cfg[key]
                if isinstance(value, Mapping):
                    value = value.get("config", {}).get("name", value.get("class_name", str(value)))
                if isinstance(value, list):
                    value = tuple(value)
                attrs.append((key, value))
        shapes = tuple(tuple(int(d) for d in w.shape) for w in layer.weights)
        sig.append((kind, tuple(attrs), shapes))
    return sig


def check_architecture(model, arch: str) -> Tuple[int, int]:
    """El modelo debe ser la arquitectura arch de errp_ae (CLAUDE.md): mismas capas en el mismo
    orden, mismos atributos (filtros, kernels, activaciones...) y formas de pesos.
    Devuelve (parámetros, MACs)."""
    reference = ae.build_model(arch, l2=0.0)
    got, expected = _layer_signature(model), _layer_signature(reference)
    if got != expected:
        diff = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), min(len(got), len(expected)))

        def show(sig, i):
            if i >= len(sig):
                return "(ninguna)"
            kind, attrs, shapes = sig[i]
            brief = ", ".join(f"{k}={v}" for k, v in attrs if k in ("filters", "units", "kernel_size", "activation",
                                                                       "depth_multiplier", "pool_size"))
            return f"{kind}({brief}) pesos {list(shapes)}"

        raise ExportError(f"el modelo no es la arquitectura '{arch}' de config.json (errp_ae.build_model): "
                          f"{len(got)} capas frente a {len(expected)}; primera diferencia en la capa {diff}: "
                          f"{show(got, diff)} en vez de {show(expected, diff)}")
    n_params = int(model.count_params())
    n_macs = int(ae.count_macs(model))
    if n_params != ae.EXPECTED_PARAMS[arch] or n_macs != ae.EXPECTED_MACS[arch]:
        raise ExportError(f"el modelo tiene {n_params} parámetros / {n_macs} MACs; '{arch}' fija "
                          f"{ae.EXPECTED_PARAMS[arch]} / {ae.EXPECTED_MACS[arch]}")
    return n_params, n_macs


def float_scores(model, z: np.ndarray) -> np.ndarray:
    """Score float32 (errp_ae.score) del modelo Keras en inferencia (BatchNorm con medias móviles)."""
    batch = PREDICT_BATCH if model.input_shape[0] is None else 1
    out = [np.zeros(0, dtype=np.float32)]
    for b in range(0, z.shape[0], batch):
        zb = z[b:b + batch]
        zh = np.asarray(model.predict_on_batch(zb[..., None]), dtype=np.float32)
        out.append(np.atleast_1d(ae.score(zb, zh)))
    s = np.concatenate(out)
    if not np.all(np.isfinite(s)):
        raise ExportError("el modelo float devuelve valores no finitos en val.npz: modelo no válido")
    return s


# ---- Conversión a TFLite int8 e inspección del .tflite ----

def convert_int8(model, rep_z: np.ndarray, tf, keras, seed: int) -> bytes:
    """PTQ int8 completa con el conjunto representativo rep_z [n, 8, 40] -> bytes del .tflite."""
    keras.utils.set_random_seed(seed)
    rep = np.ascontiguousarray(rep_z[..., None], dtype=np.float32)  # [n, 8, 40, 1]

    def representative():
        for k in range(rep.shape[0]):
            yield [rep[k:k + 1]]

    out: List[str] = []
    try:
        with captured_output(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conv = tf.lite.TFLiteConverter.from_keras_model(ae.inference_model(model))
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            buf = conv.convert()
    except Exception as e:
        log = _tail(out[0]) if out else ""
        raise ExportError(f"la conversión a TFLite int8 falló ({type(e).__name__}: {_brief(str(e))})"
                          + (f". Salida del conversor: {log}" if log else "")
                          + ". Solo se admiten capas convertibles a CONV_2D, DEPTHWISE_CONV_2D, "
                          "AVERAGE_POOL_2D, FULLY_CONNECTED, RESHAPE y RELU en int8") from None
    return bytes(buf)


@dataclass
class TfliteInfo:
    size: int
    version: int
    n_subgraphs: int
    ops: List[str]                              # en orden de ejecución
    tensor_types: List[str]
    n_inputs: int
    n_outputs: int
    in_type: str
    out_type: str
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    in_scales: List[float]
    in_zero_points: List[int]
    out_scales: List[float]
    out_zero_points: List[int]

    @property
    def in_scale(self) -> np.float32:
        return np.float32(self.in_scales[0])

    @property
    def out_scale(self) -> np.float32:
        return np.float32(self.out_scales[0])

    @property
    def in_zero_point(self) -> int:
        return int(self.in_zero_points[0])

    @property
    def out_zero_point(self) -> int:
        return int(self.out_zero_points[0])


def _enum_names(enum_class) -> Dict[int, str]:
    return {v: k for k, v in vars(enum_class).items() if not k.startswith("_") and isinstance(v, int)}


def inspect_tflite(buf: bytes) -> TfliteInfo:
    """Lee el flatbuffer con el esquema de TensorFlow (tensorflow.lite.python.schema_py_generated)."""
    try:
        from tensorflow.lite.python import schema_py_generated as fb
    except ImportError as e:
        raise ExportError(f"no se encuentra el esquema TFLite de TensorFlow ({e})") from None
    if len(buf) < 8 or buf[4:8] != b"TFL3":
        raise ExportError("el conversor no devolvió un flatbuffer TFLite (falta el identificador TFL3)")
    op_names = _enum_names(fb.BuiltinOperator)
    type_names = _enum_names(fb.TensorType)
    try:
        m = fb.Model.GetRootAsModel(buf, 0)
        n_sub = m.SubgraphsLength()
        sg = m.Subgraphs(0) if n_sub else None
        ops: List[str] = []
        types: List[str] = []
        ins: List[int] = []
        outs: List[int] = []
        if sg is not None:
            for i in range(sg.OperatorsLength()):
                code = m.OperatorCodes(sg.Operators(i).OpcodeIndex())
                builtin = max(int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode()))
                name = op_names.get(builtin, f"OP_{builtin}")
                if name == "CUSTOM":
                    name = f"CUSTOM({(code.CustomCode() or b'').decode('ascii', 'replace')})"
                ops.append(name)
            types = sorted({type_names.get(sg.Tensors(i).Type(), str(sg.Tensors(i).Type()))
                            for i in range(sg.TensorsLength())})
            ins = [int(sg.Inputs(i)) for i in range(sg.InputsLength())]
            outs = [int(sg.Outputs(i)) for i in range(sg.OutputsLength())]

        def tensor(idx_list):
            if len(idx_list) != 1:
                return "", (), [], []
            t = sg.Tensors(idx_list[0])
            q = t.Quantization()
            scales = [float(v) for v in q.ScaleAsNumpy()] if q is not None and q.ScaleLength() else []
            zps = [int(v) for v in q.ZeroPointAsNumpy()] if q is not None and q.ZeroPointLength() else []
            shape = tuple(int(d) for d in t.ShapeAsNumpy()) if t.ShapeLength() else ()
            return type_names.get(t.Type(), str(t.Type())), shape, scales, zps

        in_t, in_shape, in_s, in_zp = tensor(ins)
        out_t, out_shape, out_s, out_zp = tensor(outs)
        version = int(m.Version())
    except Exception as e:  # flatbuffer corrupto
        raise ExportError(f"no se pudo leer el .tflite generado: {type(e).__name__}: {_brief(str(e))}") from None
    return TfliteInfo(len(buf), version, n_sub, ops, types, len(ins), len(outs), in_t, out_t, in_shape, out_shape,
                      in_s, in_zp, out_s, out_zp)


def check_tflite(info: TfliteInfo) -> None:
    """Lo que exige el runner TFLM del S3 (autoencoder_tflm.cc) y el CONTRACT v3, sección 2."""
    if info.version != 3:
        raise ExportError(f"esquema TFLite {info.version}; el runner TFLM espera la versión 3")
    if info.n_subgraphs != 1:
        raise ExportError(f"el .tflite tiene {info.n_subgraphs} subgrafos; el runner TFLM exige uno")
    bad = sorted(set(info.ops) - set(ALLOWED_OPS))
    if bad:
        raise ExportError(f"el .tflite usa ops no permitidas en el S3: {', '.join(bad)} (ops del modelo: "
                          f"{', '.join(info.ops)}). Permitidas (CONTRACT v3 / CLAUDE.md): {', '.join(ALLOWED_OPS)}. "
                          "Evitar ELU, GELU, convoluciones traspuestas y capas sin kernel int8 en TFLite Micro")
    if not info.ops:
        raise ExportError("el .tflite no tiene ninguna op")
    bad_types = sorted(set(info.tensor_types) - set(ALLOWED_TENSOR_TYPES))
    if bad_types:
        raise ExportError(f"el .tflite tiene tensores {bad_types}: se esperaba un modelo int8 completo "
                          "(tensores int8 y bias/formas int32)")
    if info.n_inputs != 1 or info.n_outputs != 1:
        raise ExportError(f"el .tflite tiene {info.n_inputs} entradas y {info.n_outputs} salidas; se esperaba 1 y 1")
    for kind, ttype, shape, scales, zps in (
            ("entrada", info.in_type, info.in_shape, info.in_scales, info.in_zero_points),
            ("salida", info.out_type, info.out_shape, info.out_scales, info.out_zero_points)):
        if ttype != "INT8" or shape != IO_SHAPE:
            raise ExportError(f"{kind} del .tflite {ttype} {list(shape)}; se esperaba INT8 {list(IO_SHAPE)}")
        if len(scales) != 1 or len(zps) != 1:
            raise ExportError(f"{kind} del .tflite sin cuantización por tensor ({len(scales)} escalas, "
                              f"{len(zps)} zero points)")
        s = to_f32(scales[0])
        if not (np.isfinite(s) and s > 0 and float(s) == scales[0]):
            raise ExportError(f"{kind} del .tflite con escala no válida: {scales[0]!r}")
        if not -128 <= zps[0] <= 127:
            raise ExportError(f"{kind} del .tflite con zero point fuera de int8: {zps[0]}")


class RefInterpreter:
    """tf.lite.Interpreter con kernels de REFERENCIA (OpResolverType.BUILTIN_REF): q_in int8 [8, 40]
    -> q_out int8 [8, 40]. Igual que TFLite Micro salvo el redondeo de los empates en
    FULLY_CONNECTED (TF 2.21): ver TflmReference."""

    def __init__(self, tf, buf: bytes, info: TfliteInfo) -> None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"(?s).*tf\.lite\.Interpreter is deprecated")
                self.it = tf.lite.Interpreter(
                    model_content=buf, num_threads=1,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
            self.it.allocate_tensors()
            d = self.it.get_input_details()[0]
            o = self.it.get_output_details()[0]
        except Exception as e:
            raise ExportError(f"tf.lite.Interpreter (BUILTIN_REF) no pudo cargar el .tflite: "
                              f"{type(e).__name__}: {_brief(str(e))}") from None
        got = (float(d["quantization"][0]), int(d["quantization"][1]),
               float(o["quantization"][0]), int(o["quantization"][1]))
        expected = (info.in_scales[0], info.in_zero_points[0], info.out_scales[0], info.out_zero_points[0])
        if got != expected or tuple(d["shape"]) != IO_SHAPE or tuple(o["shape"]) != IO_SHAPE:
            raise ExportError(f"el intérprete ve otra cuantización/forma que el flatbuffer: {got} vs {expected}")
        self.in_index = d["index"]
        self.out_index = o["index"]

    def run(self, q_in: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.in_index, np.asarray(q_in, dtype=np.int8).reshape(IO_SHAPE))
        self.it.invoke()
        return np.array(self.it.get_tensor(self.out_index), dtype=np.int8).reshape(ae.N_CH, ae.N_T)


# ---- Emulación bit a bit de los kernels de referencia int8 de TFLite Micro ----
#
# tf.lite BUILTIN_REF (TF 2.21) recuantiza FULLY_CONNECTED con otro redondeo que TFLite Micro: en
# los empates difiere 1 LSB y la diferencia se propaga a las capas siguientes (medido con TFLM
# real en el PC: nivel B de test_c_engine.c). --golden-runtime tflm (por defecto) calcula q_out con
# esta emulación en enteros de esp-tflite-micro 1.4.1 (sin TFLITE_SINGLE_ROUNDING): QuantizeMultiplier
# (kernels/internal/quantization_util.cc), MultiplyByQuantizedMultiplier con doble redondeo
# (kernels/internal/common.cc, gemmlowp), multiplicadores y rangos de activación de
# kernels/kernel_util.cc y micro/kernels/*_common.cc, y los kernels
# kernels/internal/reference/integer_ops/{conv,depthwise_conv,fully_connected,pooling}.h.

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
GOLDEN_RUNTIMES = ("builtin_ref", "tflm")


def quantize_multiplier(m: float) -> Tuple[int, int]:
    """QuantizeMultiplier(double) de TFLM -> (significando Q31, shift). TfLiteRound = std::round
    (mitad lejos de cero); shift < -31 -> (0, 0)."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    v = q * float(1 << 31)  # exacto (potencia de 2)
    q_fixed = int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        shift, q_fixed = 0, 0
    return q_fixed, shift


def mul_by_quantized_multiplier(x, mult, shift) -> np.ndarray:
    """MultiplyByQuantizedMultiplier(int32 x, mult, shift) con doble redondeo (gemmlowp):
    RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x << left, mult), right). Vectorizado
    (mult y shift se difunden con x); aritmética en int64 sin desbordar."""
    x = np.asarray(x, dtype=np.int64)
    mult = np.asarray(mult, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    a = x * (np.int64(1) << left)
    if np.any((a > INT32_MAX) | (a < INT32_MIN)):
        raise ExportError("emulación TFLM: x << shift desborda int32 (comportamiento indefinido en C++)")
    ab = a * mult
    s = ab + np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    hi = np.where(s >= 0, s >> 31, -((-s) >> 31))  # división C++: trunca hacia cero
    hi = np.where((a == INT32_MIN) & (mult == INT32_MIN), np.int64(INT32_MAX), hi)
    mask = (np.int64(1) << right) - 1
    rem = hi & mask
    threshold = (mask >> 1) + (hi < 0)
    return (hi >> right) + (rem > threshold)


class TflmReference:
    """q_in int8 [8, 40] -> q_out int8 [8, 40] con la aritmética entera EXACTA de los kernels de
    referencia de TFLite Micro para las 6 ops permitidas (ver el comentario de arriba). Rechaza
    (ExportError) cualquier op o parámetro que no emule."""

    def __init__(self, buf: bytes) -> None:
        try:
            from tensorflow.lite.python import schema_py_generated as fb
        except ImportError as e:
            raise ExportError(f"no se encuentra el esquema TFLite de TensorFlow ({e})") from None
        self.fb = fb
        self.op_names = _enum_names(fb.BuiltinOperator)
        m = fb.Model.GetRootAsModel(buf, 0)
        sg = m.Subgraphs(0)
        self.tensors: List[dict] = []
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            q = t.Quantization()
            scales = (np.asarray(q.ScaleAsNumpy(), dtype=np.float32) if q is not None and q.ScaleLength()
                      else np.zeros(0, np.float32))
            zps = (np.asarray(q.ZeroPointAsNumpy(), dtype=np.int64) if q is not None and q.ZeroPointLength()
                   else np.zeros(0, np.int64))
            shape = tuple(int(d) for d in t.ShapeAsNumpy()) if t.ShapeLength() else ()
            dtype = {fb.TensorType.INT8: np.int8, fb.TensorType.INT32: np.dtype("<i4")}.get(t.Type())
            data = None
            b = m.Buffers(t.Buffer())
            if b is not None and b.DataLength():
                if dtype is None:
                    raise ExportError(f"emulación TFLM: tensor constante {i} de tipo {t.Type()} no soportado")
                data = np.frombuffer(b.DataAsNumpy().tobytes(), dtype=dtype).reshape(shape).astype(np.int64)
            elif b is not None and b.Offset() > 1:
                raise ExportError("emulación TFLM: buffers fuera del flatbuffer (modelos > 2 GB) no soportados")
            self.tensors.append({"shape": shape, "scales": scales, "zps": zps, "data": data, "type": t.Type()})
        self.input = int(sg.Inputs(0))
        self.output = int(sg.Outputs(0))
        self.plan: List[Tuple[str, Callable[[np.ndarray], np.ndarray], List[int], int]] = []
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            code = m.OperatorCodes(op.OpcodeIndex())
            name = self.op_names.get(max(int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode())), "?")
            ins = [int(op.Inputs(k)) for k in range(op.InputsLength())]
            out = int(op.Outputs(0))
            self.plan.append((name, self._prepare(name, op, ins, out), ins, out))

    # -- parámetros (Prepare de TFLM) --

    def _q(self, idx: int) -> Tuple[float, int]:
        t = self.tensors[idx]
        if t["scales"].size < 1 or t["zps"].size < 1 or t["type"] != self.fb.TensorType.INT8:
            raise ExportError(f"emulación TFLM: el tensor {idx} no es int8 cuantizado")
        return float(t["scales"][0]), int(t["zps"][0])

    def _options(self, op, cls):
        tab = op.BuiltinOptions()
        if tab is None:
            raise ExportError("emulación TFLM: op sin opciones")
        opts = cls()
        opts.Init(tab.Bytes, tab.Pos)
        return opts

    def _act_range(self, act: int, out_idx: int) -> Tuple[int, int]:
        """CalculateActivationRangeQuantized (int8): Quantize() = zp + TfLiteRound(f / scale) en float."""
        s, zp = self._q(out_idx)
        A = self.fb.ActivationFunctionType

        def quant(f: float) -> int:
            v = float(np.float32(np.float32(f) / np.float32(s)))
            return zp + int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)

        if act == A.NONE:
            return -128, 127
        if act == A.RELU:
            return max(-128, quant(0.0)), 127
        if act == A.RELU6:
            return max(-128, quant(0.0)), min(127, quant(6.0))
        if act == A.RELU_N1_TO_1:
            return max(-128, quant(-1.0)), min(127, quant(1.0))
        raise ExportError(f"emulación TFLM: activación fusionada {act} no soportada")

    def _per_channel(self, in_idx: int, w_idx: int, out_idx: int, n: int, float_product: bool
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Multiplicadores por canal: double(s_in) * double(s_w[c]) / double(s_out) (conv, depthwise
        y FC por canal); FC por tensor: double(float(s_in * s_w)) / double(s_out)."""
        s_in, _ = self._q(in_idx)
        s_out, _ = self._q(out_idx)
        ws = self.tensors[w_idx]["scales"]
        if ws.size not in (1, n):
            raise ExportError(f"emulación TFLM: {ws.size} escalas de pesos para {n} canales")
        mult, shift = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for c in range(n):
            sw = ws[c] if ws.size > 1 else ws[0]
            if float_product:
                eff = float(np.float32(np.float32(s_in) * np.float32(sw))) / float(np.float32(s_out))
            else:
                eff = float(np.float32(s_in)) * float(np.float32(sw)) / float(np.float32(s_out))
            mult[c], shift[c] = quantize_multiplier(eff)
        return mult, shift

    @staticmethod
    def _padding(padding: int, same: int, stride: int, dil: int, size: int, filt: int, out_size: int) -> int:
        eff = (filt - 1) * dil + 1
        expected = (size + stride - 1) // stride if padding == same else (size + stride - eff) // stride
        if expected != out_size:
            raise ExportError(f"emulación TFLM: tamaño de salida {out_size} != {expected}")
        return max((out_size - 1) * stride + eff - size, 0) // 2

    def _const(self, idx: int) -> Optional[np.ndarray]:
        if idx < 0:
            return None
        data = self.tensors[idx]["data"]
        if data is None:
            raise ExportError(f"emulación TFLM: el tensor {idx} debería ser constante (pesos/bias)")
        return data

    def _prepare(self, name: str, op, ins: List[int], out: int) -> Callable[[np.ndarray], np.ndarray]:
        fb = self.fb
        out_shape = self.tensors[out]["shape"]
        if name == "RESHAPE":
            return lambda x: x.reshape(out_shape)
        s_out, zp_out = self._q(out)
        s_in, zp_in = self._q(ins[0])
        if name == "RELU":
            mult, shift = quantize_multiplier(float(np.float32(np.float32(s_in) / np.float32(s_out))))
            lo = max(-128, zp_out)
            return lambda x: np.clip(zp_out + mul_by_quantized_multiplier(x - zp_in, mult, shift), lo, 127)
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            depthwise = name == "DEPTHWISE_CONV_2D"
            o = self._options(op, fb.DepthwiseConv2DOptions if depthwise else fb.Conv2DOptions)
            w = self._const(ins[1])
            bias = self._const(ins[2]) if len(ins) > 2 else None
            _, h, wd, cin = self.tensors[ins[0]]["shape"]
            _, oh, ow, cout = out_shape
            kh, kw = (w.shape[1], w.shape[2])
            sh, sw_, dh, dw = o.StrideH(), o.StrideW(), o.DilationHFactor(), o.DilationWFactor()
            ph = self._padding(o.Padding(), fb.Padding.SAME, sh, dh, h, kh, oh)
            pw = self._padding(o.Padding(), fb.Padding.SAME, sw_, dw, wd, kw, ow)
            if depthwise:
                dm = o.DepthMultiplier()
                if w.shape[0] != 1 or w.shape[3] != cout or cout != cin * dm:
                    raise ExportError("emulación TFLM: DEPTHWISE_CONV_2D con formas no soportadas")
            elif w.shape[0] != cout or w.shape[3] != cin:
                raise ExportError("emulación TFLM: CONV_2D agrupada o con formas no soportadas")
            mult, shift = self._per_channel(ins[0], ins[1], out, cout, float_product=False)
            lo, hi = self._act_range(o.FusedActivationFunction(), out)
            pad_bottom = max(0, (oh - 1) * sh - ph + dh * (kh - 1) - (h - 1))
            pad_right = max(0, (ow - 1) * sw_ - pw + dw * (kw - 1) - (wd - 1))

            def conv(x: np.ndarray) -> np.ndarray:
                # Relleno con zp_in: (zp_in + input_offset) = 0, igual que omitir los puntos de fuera
                xp = np.full((x.shape[0], ph + h + pad_bottom, pw + wd + pad_right, cin), zp_in, np.int64)
                xp[:, ph:ph + h, pw:pw + wd, :] = x
                xo = xp - zp_in
                acc = np.zeros((x.shape[0], oh, ow, cout), np.int64)
                for ky in range(kh):
                    for kx in range(kw):
                        y0, x0 = ky * dh, kx * dw
                        rows = xo[:, y0:y0 + sh * (oh - 1) + 1:sh, x0:x0 + sw_ * (ow - 1) + 1:sw_, :]
                        if depthwise:
                            acc += np.repeat(rows, cout // cin, axis=3) * w[0, ky, kx, :]
                        else:
                            acc += np.einsum("bhwc,oc->bhwo", rows, w[:, ky, kx, :])
                if bias is not None:
                    acc += bias
                return np.clip(mul_by_quantized_multiplier(acc, mult, shift) + zp_out, lo, hi)
            return conv
        if name == "FULLY_CONNECTED":
            o = self._options(op, fb.FullyConnectedOptions)
            if o.WeightsFormat() != 0:
                raise ExportError("emulación TFLM: FULLY_CONNECTED con pesos no DEFAULT")
            w = self._const(ins[1])
            bias = self._const(ins[2]) if len(ins) > 2 else None
            n_out, depth = w.shape
            ws = self.tensors[ins[1]]["scales"]
            per_channel = ws.size > 1
            mult, shift = self._per_channel(ins[0], ins[1], out, n_out, float_product=not per_channel)
            w_zps = self.tensors[ins[1]]["zps"]
            w_off = 0 if per_channel or not w_zps.size else -int(w_zps[0])  # weights_offset = -filter zp
            lo, hi = self._act_range(o.FusedActivationFunction(), out)

            def fc(x: np.ndarray) -> np.ndarray:
                xb = x.reshape(-1, depth) - zp_in
                acc = xb @ (w + w_off).T
                if bias is not None:
                    acc = acc + bias
                y = np.clip(mul_by_quantized_multiplier(acc, mult, shift) + zp_out, lo, hi)
                return y.reshape(out_shape)
            return fc
        if name == "AVERAGE_POOL_2D":
            o = self._options(op, fb.Pool2DOptions)
            _, h, wd, ch = self.tensors[ins[0]]["shape"]
            _, oh, ow, _ = out_shape
            fh, fw, sh, sw_ = o.FilterHeight(), o.FilterWidth(), o.StrideH(), o.StrideW()
            ph = self._padding(o.Padding(), fb.Padding.SAME, sh, 1, h, fh, oh)
            pw = self._padding(o.Padding(), fb.Padding.SAME, sw_, 1, wd, fw, ow)
            if (s_in, zp_in) != (s_out, zp_out):
                raise ExportError("emulación TFLM: AVERAGE_POOL_2D con cuantización distinta en entrada y salida")
            lo, hi = self._act_range(o.FusedActivationFunction(), out)

            def pool(x: np.ndarray) -> np.ndarray:
                y = np.empty((x.shape[0], oh, ow, ch), np.int64)
                for oy in range(oh):
                    for ox in range(ow):
                        iy0, ix0 = oy * sh - ph, ox * sw_ - pw
                        fy0, fy1 = max(0, -iy0), min(fh, h - iy0)
                        fx0, fx1 = max(0, -ix0), min(fw, wd - ix0)
                        n = (fy1 - fy0) * (fx1 - fx0)
                        if n <= 0:
                            raise ExportError("emulación TFLM: ventana de pooling vacía")
                        acc = x[:, iy0 + fy0:iy0 + fy1, ix0 + fx0:ix0 + fx1, :].sum(axis=(1, 2))
                        # (acc +- n/2) / n con división C++ (trunca hacia cero)
                        y[:, oy, ox, :] = np.where(acc > 0, (acc + n // 2) // n, -((n // 2 - acc) // n))
                return np.clip(y, lo, hi)
            return pool
        raise ExportError(f"emulación TFLM: op {name} no soportada")

    def run(self, q_in: np.ndarray) -> np.ndarray:
        vals: Dict[int, np.ndarray] = {self.input: np.asarray(q_in, dtype=np.int64).reshape(IO_SHAPE)}
        for name, fn, ins, out in self.plan:
            vals[out] = fn(vals[ins[0]])
        y = vals[self.output]
        if np.any(y < -128) or np.any(y > 127):
            raise ExportError("emulación TFLM: salida fuera de int8")
        return y.astype(np.int8).reshape(ae.N_CH, ae.N_T)


@dataclass
class ChainResult:
    q_in: np.ndarray                            # int8 [320]
    q_out: np.ndarray                           # int8 [320]
    z_hat: np.ndarray                           # float32 [320]
    score: float                                # float32


def int8_chain(z: np.ndarray, runner, info: TfliteInfo) -> ChainResult:
    """Cadena del motor C sobre z [8, 40] (errp_ae): quantize -> modelo int8 -> dequantize -> score."""
    zz = np.asarray(z, dtype=np.float32).reshape(ae.N_CH, ae.N_T)
    q_in = ae.quantize(zz, info.in_scale, info.in_zero_point)
    q_out = runner.run(q_in)
    z_hat = ae.dequantize(q_out, info.out_scale, info.out_zero_point)
    s = ae.score(zz, z_hat)
    return ChainResult(q_in.reshape(-1), q_out.reshape(-1), z_hat.reshape(-1), float(s))


@dataclass
class CrossCheck:
    """q_out de tf.lite BUILTIN_REF frente a la emulación de TFLM sobre los mismos q_in."""
    n_vectors: int
    n_values: int
    diff_values: int
    diff_vectors: int
    max_lsb: int

    def text(self) -> str:
        return (f"{self.diff_values} de {self.n_values} valores distintos en {self.diff_vectors} de "
                f"{self.n_vectors} vectores (max {self.max_lsb} LSB)")


def cross_check(ref: RefInterpreter, emu: TflmReference, q_ins: Sequence[np.ndarray]) -> CrossCheck:
    a = np.stack([ref.run(q) for q in q_ins]).astype(np.int64)
    b = np.stack([emu.run(q) for q in q_ins]).astype(np.int64)
    d = a != b
    return CrossCheck(len(q_ins), int(d.size), int(d.sum()), int(np.any(d, axis=(1, 2)).sum()),
                      int(np.max(np.abs(a - b))) if d.any() else 0)


# ---- Verificación float vs int8 ----

@dataclass
class Verification:
    n: int
    n_correct: int
    n_error: int
    corr: float
    corr_correct: Optional[float]
    max_abs: float
    max_rel: float
    agree: float                                # fracción de épocas con el mismo nivel
    agree_correct: Optional[float]
    agree_error: Optional[float]
    score_f: np.ndarray
    score_q: np.ndarray
    int8_thresholds: Optional[Tuple[float, float, float]]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def verify_int8(model, runner, info: TfliteInfo, cfg: ExportConfig, val: ValSet,
                min_corr: float) -> Verification:
    score_f = float_scores(model, val.z)
    score_q = np.array([int8_chain(z, runner, info).score for z in val.z], dtype=np.float32)
    if not np.all(np.isfinite(score_q)):
        raise ExportError("el modelo int8 devuelve scores no finitos en val.npz")
    corr = _pearson(score_f, score_q)
    if not math.isfinite(corr):
        raise ExportError("correlación float-int8 no definida: los scores de val.npz son constantes")
    correct = ~val.is_error
    corr_correct = _pearson(score_f[correct], score_q[correct]) if np.sum(correct) >= 3 else None
    diff = np.abs(score_f.astype(np.float64) - score_q.astype(np.float64))
    max_rel = float(np.max(diff / np.maximum(np.abs(score_f.astype(np.float64)), 1e-30)))
    lv_f = np.atleast_1d(ae.level(score_f, *cfg.thresholds))
    lv_q = np.atleast_1d(ae.level(score_q, *cfg.thresholds))
    same = lv_f == lv_q
    agree_c = float(np.mean(same[correct])) if np.any(correct) else None
    agree_e = float(np.mean(same[val.is_error])) if np.any(val.is_error) else None
    int8_t = None
    with contextlib.suppress(ValueError):
        int8_t = ae.thresholds_from_scores(score_q[correct], cfg.percentiles)
    ver = Verification(int(val.z.shape[0]), int(np.sum(correct)), int(np.sum(val.is_error)), corr,
                       corr_correct if corr_correct is None or math.isfinite(corr_correct) else None,
                       float(diff.max()), max_rel, float(np.mean(same)), agree_c, agree_e, score_f, score_q, int8_t)
    if not corr >= min_corr:
        raise ExportError(f"correlación float-int8 del score en val.npz = {corr:.7f} < --min-corr {min_corr!r} "
                          f"({ver.n} épocas; |score_f - score_q| máx {ver.max_abs:.4g}, acuerdo de nivel "
                          f"{100 * ver.agree:.1f} %): el modelo int8 no reproduce el float. Revisar rep.npz "
                          "(épocas CORRECTAS normalizadas, representativas de los datos reales). No se escribió nada")
    # Coherencia con el entrenamiento (solo avisos: los umbrales pueden haberse ajustado a mano)
    if val.score is not None:
        rel = np.abs(val.score.astype(np.float64) - score_f) / np.maximum(np.abs(score_f.astype(np.float64)), 1e-12)
        if float(np.max(rel)) > CHECK_REL:
            warn(f"val.npz 'score' difiere del score float de este modelo (hasta {100 * float(np.max(rel)):.2g} %): "
                 "¿model.keras y val.npz de entrenamientos distintos?")
    with contextlib.suppress(ValueError):
        tf_t = ae.thresholds_from_scores(score_f[correct], cfg.percentiles)
        rel_t = max(abs(a - b) / b for a, b in zip(tf_t, cfg.thresholds))
        if rel_t > CHECK_REL:
            warn("los umbrales de config.json no son los percentiles del score float de las épocas correctas de "
                 f"val.npz ({', '.join(short_float(t) for t in tf_t)}; diferencia {100 * rel_t:.2g} %): "
                 "¿umbrales ajustados a mano o archivos de entrenamientos distintos?")
    return ver


# ---- Vectores dorados ----

@dataclass
class GoldenVector:
    label: str
    epoch: np.ndarray                           # float32 [8, 40] uV: entrada de ae_infer
    z: np.ndarray                               # float32 [320]
    q_in: np.ndarray                            # int8 [320]
    q_out: np.ndarray                           # int8 [320] (runtime de --golden-runtime)
    z_hat: np.ndarray                           # float32 [320]
    score: float                                # float32
    level: int
    near: bool


@dataclass
class Goldens:
    windows: np.ndarray                         # float32 [W, 8, 250]
    win_epochs: np.ndarray                      # float32 [W, 8, 40]
    vectors: List[GoldenVector]
    cal_epochs: np.ndarray                      # float32 [C, 8, 40]
    cal_mean: np.ndarray                        # float32 [8]
    cal_std: np.ndarray                         # float32 [8]
    cal_scores: np.ndarray                      # float32 [S]
    cal_t: Tuple[float, float, float]
    base_label: str

    def counts(self) -> List[int]:
        return [sum(1 for v in self.vectors if v.level == k) for k in range(4)]

    def n_near(self) -> int:
        return sum(1 for v in self.vectors if v.near)


def golden_windows(rng: np.random.Generator) -> np.ndarray:
    """Ventanas crudas [TV_N_WIN, 8, 250] en uV (250 Hz) que ejercitan el preprocesado: offset DC
    grande + ruido; amplitudes pequeñas; deflexión tipo ErrP en Fz/Cz con un canal de offset muy
    grande (redondeo de las sumas float32)."""
    t = (np.arange(ae.WIN) - ae.PRE) * (1000.0 / ae.FS_HZ)  # ms
    w0 = rng.uniform(-300.0, 300.0, (ae.N_CH, 1)) + rng.normal(0.0, 15.0, (ae.N_CH, ae.WIN))
    w1 = rng.uniform(-2.0, 2.0, (ae.N_CH, 1)) + rng.normal(0.0, 0.5, (ae.N_CH, ae.WIN)) + 1e-3 * t[None, :]
    w2 = (rng.normal(0.0, 40.0, (ae.N_CH, 1)) + rng.normal(0.0, 8.0, (ae.N_CH, ae.WIN))
          + 5.0 * np.sin(2.0 * np.pi * 7.0 * t[None, :] / 1000.0) + rng.normal(0.0, 4.0, (ae.N_CH, 1)) * t / 1000.0)
    errp = -12.0 * np.exp(-((t - 250.0) / 40.0) ** 2) + 9.0 * np.exp(-((t - 400.0) / 60.0) ** 2)
    w2[[0, 2]] += errp[None, :]
    w2[7] += 2.0e4
    return np.stack([w0, w1, w2])[:TV_N_WIN].astype(np.float32)


def make_goldens(cfg: ExportConfig, runner, info: TfliteInfo, val: ValSet,
                 score_q: np.ndarray, seed: int) -> Goldens:
    """Vectores dorados con la calibración POR DEFECTO (MEAN_VECTOR, STD_VECTOR, THRESHOLD_LEVEL_k):
    época cero (z = 0), épocas correctas y con error de val.npz, dos vectores dirigidos por banda
    de nivel no vacía (escala de una dirección aleatoria + bisección hasta un score objetivo) y uno
    de saturación de q_in. Más goldens de preprocesado y de calibración."""
    rng = np.random.default_rng([seed, 41])
    mean, std = cfg.mean, cfg.std
    t1, t2, t3 = cfg.thresholds

    def epoch_from_z(zt: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            z32 = np.asarray(zt, dtype=np.float64).reshape(ae.N_CH, ae.N_T).astype(np.float32)
            return (mean[:, None] + z32 * std[:, None]).astype(np.float32)

    def evaluate(epoch: np.ndarray) -> Tuple[np.ndarray, ChainResult]:
        z = ae.normalize(epoch, mean, std)
        if not np.all(np.isfinite(z)):
            raise ExportError("vector dorado con z no finita (MEAN_VECTOR/STD_VECTOR extremos)")
        return z.reshape(-1), int8_chain(z, runner, info)

    def vector(label: str, epoch: np.ndarray) -> GoldenVector:
        z, r = evaluate(epoch)
        lv = ae.level(r.score, t1, t2, t3)
        near = any(abs(r.score - t) <= NEAR_REL * t for t in cfg.thresholds)
        return GoldenVector(label, epoch, z, r.q_in, r.q_out, r.z_hat, r.score, lv, near)

    vectors = [vector("cero: epoca = MEAN_VECTOR (z = 0)", np.repeat(mean[:, None], ae.N_T, axis=1))]
    correct = np.flatnonzero(~val.is_error)
    errors = np.flatnonzero(val.is_error)
    for idx, kind, n in ((correct, "correcta", TV_N_VAL_CORRECT), (errors, "con error", TV_N_VAL_ERROR)):
        if idx.size:
            for i in np.sort(rng.choice(idx, size=min(n, idx.size), replace=False)):
                vectors.append(vector(f"val.npz[{int(i)}] {kind}", epoch_from_z(val.z[i])))

    # Punto base de los dirigidos: la época correcta de val.npz con menor score int8 (lo más
    # "normal" que reconoce el modelo); sin correctas, z = 0.
    if correct.size:
        i_base = int(correct[np.argmin(score_q[correct])])
        z_base = epoch_from_z(val.z[i_base])
        z_base = ae.normalize(z_base, mean, std).astype(np.float64)
        base_label = f"val.npz[{i_base}]"
    else:
        z_base = np.zeros((ae.N_CH, ae.N_T))
        base_label = "z = 0"
    base_score = evaluate(epoch_from_z(z_base))[1].score

    def directed(level: int, target: float) -> Optional[GoldenVector]:
        for _ in range(DIRECTED_TRIES):
            d = rng.standard_normal((ae.N_CH, ae.N_T))

            def score_at(alpha: float) -> float:
                return evaluate(epoch_from_z(z_base + alpha * d))[1].score

            lo, hi = 0.0, 0.25
            s_hi = score_at(hi)
            while s_hi < target and hi < 1e6:
                lo, hi = hi, 2.0 * hi
                s_hi = score_at(hi)
            if not s_hi >= target:
                continue
            for _ in range(BISECT_ITERS):  # invariante: score(lo) < target <= score(hi)
                if hi - lo <= 1e-7 * hi:
                    break
                mid = 0.5 * (lo + hi)
                if score_at(mid) < target:
                    lo = mid
                else:
                    hi = mid
            v = vector(f"dirigido nivel {level} (score objetivo {target:.6g})", epoch_from_z(z_base + hi * d))
            if v.level == level and not v.near:
                return v
        return None

    for level, (lo_t, hi_t) in enumerate(((0.0, t1), (t1, t2), (t2, t3), (t3, math.inf))):
        if not lo_t < hi_t:
            continue  # banda vacía (umbrales iguales): sin vectores
        if hi_t <= base_score:
            raise ExportError(f"no hay vectores dorados de nivel {level}: el score int8 del punto base "
                              f"({base_label}, {short_float(base_score)}) ya supera el límite de esa banda "
                              f"({short_float(hi_t)}). ¿Umbrales de config.json incoherentes con el modelo? "
                              "(--no-test-vectors exporta sin vectores)")
        start = max(lo_t, base_score)
        if level == 0:
            targets = [start + (hi_t - start) * f for f in (0.35, 0.7)]
        elif math.isinf(hi_t):
            targets = [2.0 * start, 4.0 * start]
        else:
            targets = [start * (hi_t / start) ** f for f in (1.0 / 3.0, 2.0 / 3.0)]  # puntos geométricos
        for target in targets[:TV_PER_BAND]:
            v = directed(level, target)
            if v is None:
                raise ExportError(f"no se pudo generar un vector dorado de nivel {level} (score objetivo "
                                  f"{target:.6g}) en {DIRECTED_TRIES} direcciones (--no-test-vectors exporta sin "
                                  "vectores)")
            vectors.append(v)

    sat = vector("saturacion de q_in (z = 1000 x d)",
                 epoch_from_z(SATURATION_Z * rng.standard_normal((ae.N_CH, ae.N_T))))
    if not (np.any(sat.q_in == 127) and np.any(sat.q_in == -128)):
        raise ExportError("el vector de saturación no satura q_in (escala de entrada enorme?)")
    vectors.append(sat)
    for v in vectors:
        if not (np.all(np.isfinite(v.epoch)) and np.all(np.isfinite(v.z_hat)) and math.isfinite(v.score)):
            raise ExportError(f"vector dorado no finito: {v.label}")

    windows = golden_windows(rng)
    win_epochs = ae.preprocess_window(windows)

    # Calibración: épocas de val.npz (o sintéticas) con OTRA media/escala por canal -> mean/std;
    # scores sintéticos alrededor de T1 -> umbrales con los percentiles del modelo
    if correct.size >= TV_CAL_N_EPOCHS:
        pool = val.z[correct]
    else:
        pool = rng.standard_normal((TV_CAL_N_EPOCHS, ae.N_CH, ae.N_T))
    pick = np.sort(rng.choice(pool.shape[0], size=TV_CAL_N_EPOCHS, replace=False))
    cal_mean_true = mean + rng.uniform(-5.0, 5.0, ae.N_CH)
    cal_std_true = std * rng.uniform(0.8, 1.25, ae.N_CH)
    cal_epochs = (cal_mean_true[None, :, None] + pool[pick].astype(np.float64) * cal_std_true[None, :, None])
    cal_epochs = cal_epochs.astype(np.float32)
    try:
        cal_mean, cal_std = ae.norm_stats(cal_epochs)
        cal_scores = (t1 * np.exp(rng.normal(-0.2, 0.25, TV_CAL_N_SCORES))).astype(np.float32)
        cal_t = ae.thresholds_from_scores(cal_scores, cfg.percentiles)
    except ValueError as e:
        raise ExportError(f"goldens de calibración: {e}") from None
    return Goldens(windows, win_epochs, vectors, cal_epochs, cal_mean, cal_std, cal_scores, cal_t, base_label)


# ---- Identidad del modelo ----

def compute_model_id(tflite: bytes, cfg: ExportConfig) -> str:
    """16 hex del sha256 de: bytes del .tflite + float32 little-endian de MEAN_VECTOR[8],
    STD_VECTOR[8], THRESHOLD_LEVEL_1..3 y AE_CALIB_PCT_1..3."""
    h = hashlib.sha256(tflite)
    for values in (cfg.mean, cfg.std, cfg.thresholds, cfg.percentiles):
        h.update(np.asarray(values, dtype="<f4").tobytes())
    return h.hexdigest()[:16]


# ---- Generación de los headers ----

@dataclass
class Provenance:
    files: List[Tuple[str, str, str]]           # (qué, nombre base ASCII, sha256)
    command: List[str]                          # comando de regeneración en trozos (líneas <= 120)
    tf_version: str
    keras_version: str


def _rows(values: Sequence[str], indent: str, per_line: int) -> List[str]:
    return [indent + ", ".join(values[i:i + per_line]) + "," for i in range(0, len(values), per_line)]


def _float_rows(values: np.ndarray, indent: str) -> List[str]:
    return _rows([c_float(v) for v in np.asarray(values).ravel()], indent, VALUES_PER_LINE)


def _int_rows(values: np.ndarray, indent: str) -> List[str]:
    return _rows([str(int(v)) for v in np.asarray(values).ravel()], indent, INTS_PER_LINE)


def _c_array(decl: str, arr: np.ndarray, rows: Callable[[np.ndarray, str], List[str]],
             labels: Optional[Sequence[str]] = None) -> List[str]:
    """static const <tipo> NOMBRE[d0][d1]... con llaves anidadas (sin -Wmissing-braces)."""
    arr = np.asarray(arr)
    lines = [f"{decl} = {{"]

    def rec(a: np.ndarray, depth: int, label: Optional[str]) -> List[str]:
        ind = "    " * (depth + 1)
        head = ind + "{" + (f" // {label}" if label else "")
        if a.ndim == 1:
            return [head] + rows(a, ind + "    ") + [ind + "},"]
        out = [head]
        for sub in a:
            out += rec(sub, depth + 1, None)
        return out + [ind + "},"]

    if arr.ndim == 1:
        lines += rows(arr, "    ")
    else:
        for i, sub in enumerate(arr):
            lines += rec(sub, 0, labels[i] if labels else None)
    lines.append("};")
    return lines


def _ops_text(ops: Sequence[str]) -> str:
    """CONV_2D, ..., FULLY_CONNECTED x3, RESHAPE (repeticiones seguidas agrupadas)."""
    parts: List[str] = []
    i = 0
    while i < len(ops):
        j = i
        while j + 1 < len(ops) and ops[j + 1] == ops[i]:
            j += 1
        parts.append(ops[i] + (f" x{j - i + 1}" if j > i else ""))
        i = j + 1
    return ", ".join(parts)


def corr_text(corr: float) -> str:
    """Correlación truncada (nunca redondeada hacia arriba) a 6 decimales: texto estable entre
    máquinas aunque los floats de Keras difieran en los últimos bits."""
    return f"{math.floor(corr * 1e6) / 1e6:.6f}"


def runtime_text(golden_runtime: str, tf_version: str) -> str:
    if golden_runtime == "tflm":
        return "emulacion entera bit a bit de los kernels de referencia int8 de TFLite Micro (esp-tflite-micro 1.4.1)"
    return f"tf.lite.Interpreter con kernels de REFERENCIA (OpResolverType.BUILTIN_REF, TF {tf_version})"


def _wrap_comment(text: str, prefix: str = "// ", width: int = 118) -> List[str]:
    """Parte un texto en líneas de comentario // de como mucho width caracteres."""
    lines: List[str] = []
    cur = ""
    for word in text.split():
        if cur and len(prefix) + len(cur) + 1 + len(word) > width:
            lines.append(prefix + cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(prefix + cur)
    return lines


def _xcheck_lines(golden_runtime: str, xc: Optional[CrossCheck], what: str) -> List[str]:
    if xc is None:
        return _wrap_comment(f"Contraste BUILTIN_REF / emulacion de TFLM no disponible para {what}.")
    if golden_runtime == "tflm":
        text = (f"Contraste en {what}: tf.lite BUILTIN_REF difiere de estos q_out en {xc.text()}; es el redondeo de "
                "los empates en FULLY_CONNECTED (TF 2.21 frente a TFLM).")
    else:
        text = (f"Contraste en {what}: la emulacion de TFLite Micro (--golden-runtime tflm) difiere de estos q_out "
                f"en {xc.text()}; es el redondeo de los empates en FULLY_CONNECTED. Con TFLM real (nivel B) esas "
                "posiciones salen distintas: para goldens bit a bit con TFLM, regenerar con --golden-runtime tflm.")
    return _wrap_comment(text)


def render_weights_header(model_id: str, cfg: ExportConfig, info: TfliteInfo, tflite: bytes, n_params: int,
                          n_macs: int, ver: Verification, arena_bytes: int, arena_default: bool,
                          n_rep: int, prov: Provenance, golden_runtime: str) -> str:
    out: List[str] = []
    add = out.append
    pk = [ae.percentile_key(p) for p in cfg.percentiles]
    add("#ifndef AUTOENCODER_WEIGHTS_H")
    add("#define AUTOENCODER_WEIGHTS_H")
    add("// GENERADO por ml/c_exporter/export_to_c.py - NO EDITAR A MANO.")
    add("//")
    add("// ErrP-AE int8 del detector C2 para TensorFlow Lite Micro (CONTRACT v3, seccion 6) y sus")
    add("// metadatos. Solo lo incluye src/autoencoder_tflm.cc (una sola copia del modelo en flash); el")
    add("// test de nivel A (test_c_engine.c con -DAE_TEST_FAKE_RUNNER) lo incluye para su runner falso.")
    add("//")
    for what, name, sha in prov.files:
        add(f"// {what:<7} {name} (sha256 {sha})")
    add(f"// Arquitectura: {cfg.arch} (ml/autoencoder/errp_ae.py), {n_params} parametros, {n_macs} MACs por epoca")
    add(f"// Ops TFLite ({len(info.ops)}): {_ops_text(info.ops)}")
    add(f"// Conversion: TF {prov.tf_version} / Keras {prov.keras_version}, tf.lite.TFLiteConverter, PTQ int8")
    add(f"//   completa (TFLITE_BUILTINS_INT8) con {n_rep} epocas de rep.npz, entrada y salida int8")
    add(f"//   {list(IO_SHAPE)} [electrodo][tiempo], cuantizacion por tensor; .tflite de {info.size} bytes.")
    add(f"// Float vs int8 (score MSE) en {ver.n} epocas de val.npz ({ver.n_correct} correctas + {ver.n_error} "
        "con error):")
    add(f"//   correlacion de Pearson {corr_text(ver.corr)} (CLAUDE.md: >= 0.98); |score_f - score_q| max "
        f"{ver.max_abs:.3g};")
    add(f"//   acuerdo de nivel {100 * ver.agree:.1f} % con los umbrales por defecto. Score int8: cadena del motor C")
    out.extend(_wrap_comment(f"(errp_ae) con {runtime_text(golden_runtime, prov.tf_version)}.", "//   "))
    add("//")
    add("// Cadena en el S3: z = (epoca - MEAN_VECTOR[c]) / STD_VECTOR[c];")
    add("//   q_in = clamp(lrintf(z / AE_IN_SCALE) + AE_IN_ZERO_POINT, -128, 127); q_out = modelo(q_in);")
    add("//   z_hat = AE_OUT_SCALE * (float)(q_out - AE_OUT_ZERO_POINT); score = (1/320) * sum (z - z_hat)^2;")
    add("//   nivel 0 si score <= T1; 1 si <= T2; 2 si <= T3; 3 si score > T3 (\">\" estricto) o fail-safe.")
    add("// MEAN_VECTOR / STD_VECTOR / THRESHOLD_LEVEL_k son la calibracion POR DEFECTO (config.json): en el")
    add("// S3 se recalibra por operador y sesion (AE_CALIB_PCT_k = percentiles de esa calibracion).")
    add("// AE_MODEL_ID: primeros 16 hex del sha256 de los bytes de AE_MODEL_TFLITE seguidos de los float32")
    add("// little-endian de MEAN_VECTOR, STD_VECTOR, THRESHOLD_LEVEL_1..3 y AE_CALIB_PCT_1..3.")
    if arena_default:
        out.extend(_wrap_comment(f"AE_TENSOR_ARENA_BYTES: ESTIMACION por defecto del exportador ({ARENA_NOTE}; "
                                 "con margen para los kernels ESP-NN). Medir la arena usada (niveles B/C de "
                                 "test_c_engine.c la imprimen) y regenerar con --arena-bytes."))
    else:
        add("// AE_TENSOR_ARENA_BYTES: valor de --arena-bytes (arena medida en un build TFLM + margen).")
    if cfg.placeholder:
        add("// PLACEHOLDER: pesos de prueba (fixture o datos sinteticos), NO un modelo entrenado con EEG real.")
    add("//")
    add("// Regenerar desde la raiz del repo (reescribe tambien firmware/detector_s3/test/autoencoder_test_vectors.h):")
    for k, part in enumerate(prov.command):
        add(("//   " if k == 0 else "//       ") + part)
    add("")
    add(f'#define AE_MODEL_ID "{model_id}"')
    add(f'#define AE_MODEL_ARCH "{cfg.arch}"')
    add(f"#define AE_WEIGHTS_PLACEHOLDER {1 if cfg.placeholder else 0} "
        "// 1 = pesos de prueba / datos sinteticos, NO un modelo entrenado con EEG real")
    add("")
    add("#define INPUT_DIM 320")
    add("#define OUTPUT_DIM 320")
    add(f"#define AE_MODEL_N_CH {ae.N_CH}")
    add(f"#define AE_MODEL_N_T {ae.N_T}")
    add("")
    add(f"#define AE_IN_SCALE {c_float(info.in_scale)}")
    add(f"#define AE_IN_ZERO_POINT {c_int(info.in_zero_point)}")
    add(f"#define AE_OUT_SCALE {c_float(info.out_scale)}")
    add(f"#define AE_OUT_ZERO_POINT {c_int(info.out_zero_point)}")
    add(f"#define AE_TENSOR_ARENA_BYTES {arena_bytes}")
    add(f"#define AE_INT8_SCORE_CORR {corr_text(ver.corr)}f")
    add(f"#define AE_MODEL_N_PARAMS {n_params}")
    add(f"#define AE_MODEL_N_MACS {n_macs}")
    add("")
    for k, (t, key) in enumerate(zip(cfg.thresholds, pk), 1):
        add(f"#define THRESHOLD_LEVEL_{k} {c_float(t)} // default T{k} = {key} ({LEVEL_NAMES[k - 1]})")
    for k, p in enumerate(cfg.percentiles, 1):
        add(f"#define AE_CALIB_PCT_{k} {c_float(p)}")
    add("")
    add(f"// Calibracion por defecto por canal ({', '.join(ae.CHANNELS)}): z = (x - MEAN) / STD")
    out.extend(_c_array("static const float MEAN_VECTOR[AE_MODEL_N_CH]", cfg.mean, _float_rows))
    out.extend(_c_array("static const float STD_VECTOR[AE_MODEL_N_CH]", cfg.std, _float_rows))
    add("")
    add("// Modelo TFLite int8 (flatbuffer TFL3), alineado a 16 bytes para TFLite Micro")
    add("static const unsigned char AE_MODEL_TFLITE[] __attribute__((aligned(16))) = {")
    out.extend(_rows([f"0x{b:02x}" for b in tflite], "    ", BYTES_PER_LINE))
    add("};")
    add(f"static const unsigned int AE_MODEL_TFLITE_LEN = {len(tflite)};")
    add("")
    add("#endif // AUTOENCODER_WEIGHTS_H")
    return "\n".join(out) + "\n"


def render_test_vectors(model_id: str, cfg: ExportConfig, g: Goldens, prov: Provenance, seed: int,
                        golden_runtime: str, xc: Optional[CrossCheck]) -> str:
    counts = g.counts()
    labels = [ascii_comment(f"[{i}] {v.label}", 80) for i, v in enumerate(g.vectors)]
    out: List[str] = []
    add = out.append
    add("#ifndef AUTOENCODER_TEST_VECTORS_H")
    add("#define AUTOENCODER_TEST_VECTORS_H")
    add("// GENERADO por ml/c_exporter/export_to_c.py - NO EDITAR A MANO.")
    add("//")
    add("// Vectores dorados de firmware/detector_s3/test/test_c_engine.c (CONTRACT v3, secciones 6 y 8),")
    add("// generados junto con include/autoencoder_weights.h (mismo AE_MODEL_ID: si no coinciden,")
    add("// regenerar los dos). Referencia numerica: ml/autoencoder/errp_ae.py (float32, sumas en orden")
    add(f"// de indice, sin FMA). Semilla de los vectores: {seed}.")
    for what, name, sha in prov.files:
        add(f"//   {what:<7} {name} (sha256 {sha})")
    add("//")
    add("// Preprocesado: AE_TV_WIN (ventanas crudas [8][250] en uV) -> AE_TV_WIN_EPOCH =")
    add("//   errp_ae.preprocess_window (= ae_preprocess, bit a bit).")
    add("// Inferencia con la calibracion POR DEFECTO (MEAN_VECTOR, STD_VECTOR, THRESHOLD_LEVEL_k):")
    add("//   AE_TV_EPOCH  epoca [8][40] en uV (entrada de ae_infer)")
    add("//   AE_TV_Z      (epoca - MEAN) / STD en float32 ([canal][tiempo], 320)")
    add("//   AE_TV_Q_IN   clamp(rint(z / AE_IN_SCALE) + AE_IN_ZERO_POINT, -128, 127)")
    add("//   AE_TV_Q_OUT  salida int8 del modelo con:")
    out.extend(_wrap_comment(runtime_text(golden_runtime, prov.tf_version), "//                "))
    add("//   AE_TV_ZHAT   AE_OUT_SCALE * (float)(q_out - AE_OUT_ZERO_POINT)")
    add("//   AE_TV_SCORE  (1/320) * sum (z - z_hat)^2 en float32, acumulado en orden de indice")
    add("//   AE_TV_LEVEL  0..3 con \">\" estricto; AE_TV_NEAR_THRESHOLD = 1 si |score - Tk| <= 1e-4 * Tk (el")
    add("//                test no exige el nivel exacto de ese vector)")
    out.extend(_xcheck_lines(golden_runtime, xc, "estos vectores"))
    add("// Cobertura: epoca cero (z = 0), epocas correctas y con error de val.npz, 2 vectores dirigidos por")
    add(f"// banda de nivel no vacia (escala de una direccion aleatoria desde {ascii_comment(g.base_label)}, la")
    add("// correcta de menor score, + biseccion hasta un score objetivo) y uno de saturacion de q_in.")
    add(f"// Vectores por nivel 0/1/2/3: {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}; cerca de un umbral: "
        f"{g.n_near()}.")
    add("// Calibracion: AE_TV_CAL_EPOCHS -> AE_TV_CAL_MEAN / AE_TV_CAL_STD (errp_ae.norm_stats: float64 ->")
    add("//   float32, ddof 0); AE_TV_CAL_SCORES -> AE_TV_CAL_T = percentiles \"linear\" AE_CALIB_PCT_1..3")
    add("//   (errp_ae.thresholds_from_scores). El C calcula en float32: comparar con tolerancia.")
    add("")
    add("#include <stdint.h>")
    add("")
    add(f'#define AE_TV_MODEL_ID "{model_id}"')
    add(f"#define AE_TV_N_WIN {g.windows.shape[0]}")
    add(f"#define AE_TV_COUNT {len(g.vectors)}")
    add(f"#define AE_TV_CAL_N_EPOCHS {g.cal_epochs.shape[0]}")
    add(f"#define AE_TV_CAL_N_SCORES {g.cal_scores.size}")
    add("")
    add("// Vectores: indice, origen, score, nivel (* = cerca de un umbral)")
    for i, v in enumerate(g.vectors):
        add(f"//   {labels[i]}: score {short_float(v.score)}, nivel {v.level}{' *' if v.near else ''}")
    add("")
    win_labels = [f"ventana {w}" for w in range(g.windows.shape[0])]
    out.extend(_c_array(f"static const float AE_TV_WIN[AE_TV_N_WIN][{ae.N_CH}][{ae.WIN}]", g.windows, _float_rows,
                        win_labels))
    add("")
    out.extend(_c_array(f"static const float AE_TV_WIN_EPOCH[AE_TV_N_WIN][{ae.N_CH}][{ae.N_T}]", g.win_epochs,
                        _float_rows, win_labels))
    add("")
    out.extend(_c_array(f"static const float AE_TV_EPOCH[AE_TV_COUNT][{ae.N_CH}][{ae.N_T}]",
                        np.stack([v.epoch for v in g.vectors]), _float_rows, labels))
    add("")
    idx = [f"[{i}]" for i in range(len(g.vectors))]
    out.extend(_c_array(f"static const float AE_TV_Z[AE_TV_COUNT][{ae.N_IN}]", np.stack([v.z for v in g.vectors]),
                        _float_rows, idx))
    add("")
    out.extend(_c_array(f"static const int8_t AE_TV_Q_IN[AE_TV_COUNT][{ae.N_IN}]",
                        np.stack([v.q_in for v in g.vectors]), _int_rows, idx))
    add("")
    out.extend(_c_array(f"static const int8_t AE_TV_Q_OUT[AE_TV_COUNT][{ae.N_IN}]",
                        np.stack([v.q_out for v in g.vectors]), _int_rows, idx))
    add("")
    out.extend(_c_array(f"static const float AE_TV_ZHAT[AE_TV_COUNT][{ae.N_IN}]",
                        np.stack([v.z_hat for v in g.vectors]), _float_rows, idx))
    add("")
    out.extend(_c_array("static const float AE_TV_SCORE[AE_TV_COUNT]",
                        np.array([v.score for v in g.vectors], dtype=np.float32), _float_rows))
    add("")
    out.extend(_c_array("static const int AE_TV_LEVEL[AE_TV_COUNT]", np.array([v.level for v in g.vectors]),
                        _int_rows))
    add("")
    out.extend(_c_array("static const unsigned char AE_TV_NEAR_THRESHOLD[AE_TV_COUNT]",
                        np.array([1 if v.near else 0 for v in g.vectors]), _int_rows))
    add("")
    out.extend(_c_array(f"static const float AE_TV_CAL_EPOCHS[AE_TV_CAL_N_EPOCHS][{ae.N_CH}][{ae.N_T}]",
                        g.cal_epochs, _float_rows, [f"epoca de calibracion {e}" for e in range(g.cal_epochs.shape[0])]))
    add("")
    out.extend(_c_array(f"static const float AE_TV_CAL_MEAN[{ae.N_CH}]", g.cal_mean, _float_rows))
    out.extend(_c_array(f"static const float AE_TV_CAL_STD[{ae.N_CH}]", g.cal_std, _float_rows))
    add("")
    out.extend(_c_array("static const float AE_TV_CAL_SCORES[AE_TV_CAL_N_SCORES]", g.cal_scores, _float_rows))
    add("// p" + " / p".join(format(p, "g") for p in cfg.percentiles) + " de AE_TV_CAL_SCORES")
    out.extend(_c_array("static const float AE_TV_CAL_T[3]", np.array(g.cal_t, dtype=np.float32), _float_rows))
    add("")
    add("#endif // AUTOENCODER_TEST_VECTORS_H")
    return "\n".join(out) + "\n"


def regen_command(names: Dict[str, str], min_corr: float, seed: int, arena_bytes: Optional[int],
                  golden_runtime: str) -> List[str]:
    """Comando de regeneración (solo nombres base) partido en trozos para el comentario."""
    parts = [f"python {SCRIPT_REL} --model <ruta>/{names['model']} --config <ruta>/{names['config']}",
             f"--rep <ruta>/{names['rep']} --val <ruta>/{names['val']}"]
    extra = []
    if min_corr != DEFAULT_MIN_CORR:
        extra.append(f"--min-corr {min_corr!r}")
    if seed != 0:
        extra.append(f"--seed {seed}")
    if arena_bytes is not None:
        extra.append(f"--arena-bytes {arena_bytes}")
    if golden_runtime != DEFAULT_GOLDEN_RUNTIME:
        extra.append(f"--golden-runtime {golden_runtime}")
    if extra:
        parts.append(" ".join(extra))
    return parts


# ---- Exportación completa ----

@dataclass
class ExportResult:
    model_id: str
    cfg: ExportConfig
    tflite: bytes
    info: TfliteInfo
    n_params: int
    n_macs: int
    n_rep: int
    ver: Verification
    goldens: Optional[Goldens]
    weights_text: str
    vectors_text: Optional[str]
    arena_bytes: int
    arena_default: bool
    golden_runtime: str
    xcheck_val: Optional[CrossCheck]            # BUILTIN_REF vs emulación TFLM en val.npz
    xcheck_tv: Optional[CrossCheck]             # ... en los vectores dorados
    out_path: Path
    tv_path: Optional[Path]
    tflite_out: Optional[Path]


def check_paths(inputs: Dict[str, Path], outputs: Dict[str, Optional[Path]]) -> None:
    """Salidas: no directorios, distintas entre sí y de las entradas."""
    real_out = {k: Path(v) for k, v in outputs.items() if v is not None}
    for opt, path in real_out.items():
        if path.is_dir():
            raise ExportError(f"{opt} apunta a un directorio: {path}")
    names = list(real_out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _same_file(real_out[a], real_out[b]):
                raise ExportError(f"{a} y {b} apuntan al mismo archivo: {real_out[a]}")
        for opt, path in inputs.items():
            if _same_file(real_out[a], path):
                raise ExportError(f"{a} sobrescribiría la entrada {opt}: {path}")


def run_export(model_path: Path, config_path: Path, rep_path: Path, val_path: Path,
               out_path: Path = DEFAULT_WEIGHTS, tv_path: Optional[Path] = DEFAULT_TEST_VECTORS,
               tflite_out: Optional[Path] = None, min_corr: float = DEFAULT_MIN_CORR, seed: int = 0,
               arena_bytes: Optional[int] = None, golden_runtime: str = DEFAULT_GOLDEN_RUNTIME) -> ExportResult:
    """Valida, convierte, verifica y SOLO al final escribe las salidas (atómicamente)."""
    model_path, config_path, rep_path, val_path = (Path(p) for p in (model_path, config_path, rep_path, val_path))
    out_path = Path(out_path)
    tv_path = Path(tv_path) if tv_path is not None else None
    tflite_out = Path(tflite_out) if tflite_out is not None else None
    inputs = {"--model": model_path, "--config": config_path, "--rep": rep_path, "--val": val_path}
    check_paths(inputs, {"--out": out_path, "--test-vectors": tv_path, "--tflite-out": tflite_out})
    if not (math.isfinite(min_corr) and 0.0 <= min_corr <= 1.0):
        raise ExportError(f"--min-corr debe estar en [0, 1] (es {min_corr!r})")
    if arena_bytes is not None and not ARENA_MIN <= int(arena_bytes) <= ARENA_MAX:
        raise ExportError(f"--arena-bytes debe estar entre {ARENA_MIN} y {ARENA_MAX}")
    if golden_runtime not in GOLDEN_RUNTIMES:
        raise ExportError(f"--golden-runtime {golden_runtime!r} no válido (opciones: {', '.join(GOLDEN_RUNTIMES)})")
    for opt, path in inputs.items():
        if not path.is_file():
            raise ExportError(f"no existe {opt}: {path}")

    cfg, cfg_bytes = load_config(config_path)
    rep_arrays, rep_bytes = load_npz(rep_path, "el conjunto representativo (rep.npz)")
    rep_z = parse_rep(rep_arrays, rep_path.name)
    val_arrays, val_bytes = load_npz(val_path, "el conjunto de validación (val.npz)")
    val = parse_val(val_arrays, val_path.name)

    tf, keras = import_tf()
    model = load_keras_model(model_path, keras)
    model_bytes = model_path.read_bytes()
    tflite = convert_int8(model, rep_z, tf, keras, seed)
    info = inspect_tflite(tflite)
    check_tflite(info)
    n_params, n_macs = check_architecture(model, cfg.arch)
    ref = RefInterpreter(tf, tflite, info)
    emu: Optional[TflmReference] = None
    try:
        emu = TflmReference(tflite)
    except ExportError as e:
        if golden_runtime == "tflm":
            raise
        warn(f"sin contraste con la emulación de TFLM ({e})")
    runner = emu if golden_runtime == "tflm" else ref
    ver = verify_int8(model, runner, info, cfg, val, min_corr)
    q_val = [ae.quantize(z, info.in_scale, info.in_zero_point) for z in val.z]
    xc_val = cross_check(ref, emu, q_val) if emu is not None else None
    if xc_val is not None and (xc_val.max_lsb > XCHECK_MAX_LSB
                               or xc_val.diff_values > XCHECK_MAX_FRAC * xc_val.n_values):
        msg = (f"la emulación de TFLM no cuadra con BUILTIN_REF en val.npz ({xc_val.text()}): se esperan solo "
               "diferencias sueltas de 1 LSB (redondeo de FULLY_CONNECTED)")
        if golden_runtime == "tflm":
            raise ExportError(msg + "; no se generan goldens con ella")
        warn(msg)

    model_id = compute_model_id(tflite, cfg)
    names = {"model": ascii_comment(model_path.name, 36), "config": ascii_comment(config_path.name, 36),
             "rep": ascii_comment(rep_path.name, 36), "val": ascii_comment(val_path.name, 36)}
    prov = Provenance(
        [("Modelo:", names["model"], sha256_bytes(model_bytes)), ("Config:", names["config"], sha256_bytes(cfg_bytes)),
         ("Rep:", names["rep"], sha256_bytes(rep_bytes)), ("Val:", names["val"], sha256_bytes(val_bytes))],
        regen_command(names, min_corr, seed, arena_bytes, golden_runtime), ascii_comment(tf.__version__, 32),
        ascii_comment(keras.__version__, 32))
    arena = int(arena_bytes) if arena_bytes is not None else DEFAULT_ARENA_BYTES
    weights_text = render_weights_header(model_id, cfg, info, tflite, n_params, n_macs, ver, arena,
                                         arena_bytes is None, int(rep_z.shape[0]), prov, golden_runtime)
    goldens = None
    vectors_text = None
    xc_tv = None
    if tv_path is not None:
        goldens = make_goldens(cfg, runner, info, val, ver.score_q, seed)
        if emu is not None:
            xc_tv = cross_check(ref, emu, [v.q_in for v in goldens.vectors])
            if golden_runtime == "builtin_ref" and xc_tv.diff_values:
                warn(f"q_out dorados de BUILTIN_REF frente a la emulación de TFLite Micro: {xc_tv.text()} "
                     "(redondeo de FULLY_CONNECTED). El nivel B de test_c_engine.c (TFLM real) verá esas diferencias; "
                     "--golden-runtime tflm genera goldens bit a bit con TFLM")
        vectors_text = render_test_vectors(model_id, cfg, goldens, prov, seed, golden_runtime, xc_tv)
    else:
        warn("--no-test-vectors: no se generan vectores dorados; un autoencoder_test_vectors.h de otro modelo "
             "hará fallar test_c_engine (AE_TV_MODEL_ID distinto) hasta regenerarlo")

    items: List[Tuple[Path, bytes]] = []
    if tflite_out is not None:
        items.append((tflite_out, tflite))
    items.append((out_path, weights_text.encode("ascii")))
    if tv_path is not None and vectors_text is not None:
        items.append((tv_path, vectors_text.encode("ascii")))
    write_files_atomic(items)
    return ExportResult(model_id, cfg, tflite, info, n_params, n_macs, int(rep_z.shape[0]), ver, goldens,
                        weights_text, vectors_text, arena, arena_bytes is None, golden_runtime, xc_val, xc_tv,
                        out_path, tv_path, tflite_out)


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100.0 * v:.1f} %"


def print_summary(r: ExportResult) -> None:
    cfg, info, ver = r.cfg, r.info, r.ver
    pk = [ae.percentile_key(p) for p in cfg.percentiles]
    print("Exportación completada (ErrP-AE int8 para TFLite Micro):")
    print(f"  arquitectura    : {cfg.arch} ({r.n_params} parámetros, {r.n_macs} MACs por época)")
    print(f"  .tflite         : {info.size} bytes; ops: {_ops_text(info.ops)} (todas permitidas)")
    print(f"  cuantización    : entrada s={short_float(info.in_scale)} zp={info.in_zero_point} | salida "
          f"s={short_float(info.out_scale)} zp={info.out_zero_point} (int8 {list(IO_SHAPE)})")
    print(f"  conjunto PTQ    : {r.n_rep} épocas de rep.npz")
    rt = "emulación de TFLite Micro" if r.golden_runtime == "tflm" else "tf.lite BUILTIN_REF"
    cc = "" if ver.corr_correct is None else f"; solo correctas {corr_text(ver.corr_correct)}"
    print(f"  float vs int8   : correlación {corr_text(ver.corr)} (mínimo {DEFAULT_MIN_CORR:g} del CLAUDE.md){cc}; "
          f"{ver.n} épocas de val.npz ({ver.n_correct} correctas + {ver.n_error} con error); int8 con {rt}")
    print(f"                    |score_f - score_q| máx {ver.max_abs:.4g} (relativo {100 * ver.max_rel:.2f} %)")
    print(f"  acuerdo de nivel: {_pct(ver.agree)} (correctas {_pct(ver.agree_correct)}, con error "
          f"{_pct(ver.agree_error)}) con los umbrales por defecto")
    print("  umbrales        : " + "  ".join(f"{k}={short_float(t)}" for k, t in zip(pk, cfg.thresholds))
          + " (config.json)")
    if ver.int8_thresholds is not None:
        print("                    " + "  ".join(f"{k}={short_float(t)}" for k, t in zip(pk, ver.int8_thresholds))
              + " (informativo: percentiles del score int8 de las correctas de val.npz)")
    print(f"  AE_MODEL_ID     : {r.model_id}")
    if r.arena_default:
        print(f"  arena TFLM      : {r.arena_bytes} B (ESTIMACIÓN por defecto: {ARENA_NOTE}; medir con el nivel B/C "
              "de test_c_engine.c y pasar --arena-bytes)")
    else:
        print(f"  arena TFLM      : {r.arena_bytes} B (--arena-bytes)")
    if cfg.placeholder:
        print("  placeholder     : SÍ -> AE_WEIGHTS_PLACEHOLDER 1 (NO es un modelo entrenado con EEG real)")
    else:
        print("  placeholder     : no -> AE_WEIGHTS_PLACEHOLDER 0")
    if r.xcheck_val is not None:
        print(f"  BUILTIN_REF vs emulación TFLM: val.npz {r.xcheck_val.text()}")
    print(f"  pesos           : {r.out_path}")
    if r.goldens is not None:
        c = r.goldens.counts()
        print(f"  vectores        : {r.tv_path}")
        print(f"                    {len(r.goldens.vectors)} épocas (q_out con {rt}; por nivel 0/1/2/3: "
              f"{c[0]}/{c[1]}/{c[2]}/{c[3]}; cerca de un umbral: {r.goldens.n_near()}), "
              f"{r.goldens.windows.shape[0]} ventanas, calibración {r.goldens.cal_epochs.shape[0]} épocas + "
              f"{r.goldens.cal_scores.size} scores")
        if r.xcheck_tv is not None:
            print(f"                    contraste BUILTIN_REF vs emulación TFLM en los goldens: {r.xcheck_tv.text()}")
    else:
        print("  vectores        : no generados (--no-test-vectors)")
    if r.tflite_out is not None:
        print(f"  .tflite         : {r.tflite_out}")


# ---- Fixture (modelo aleatorio de prueba, sin entrenar) ----

def _gauss(t: np.ndarray, center: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t[None, :] - np.asarray(center)[:, None]) / width) ** 2)


def fixture_windows(rng: np.random.Generator, n_correct: int, n_error: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ventanas crudas sintéticas [n, 8, 250] en uV (250 Hz) + is_error: osciladores 2-20 Hz,
    ruido de sensor, deriva lenta, offset DC por canal y época, respuesta centro-parietal ~300 ms
    en todas y, en las de error, deflexión tipo ErrP en Fz/Cz (negatividad ~250 ms, positividad
    ~400 ms). Solo para probar la tubería: no es EEG real."""
    n = n_correct + n_error
    t = (np.arange(ae.WIN) - ae.PRE) / float(ae.FS_HZ)  # s
    err = np.zeros(n, dtype=bool)
    err[:n_error] = True
    rng.shuffle(err)
    x = np.zeros((n, ae.N_CH, ae.WIN))
    for _ in range(4):
        f = rng.uniform(2.0, 20.0, (n, ae.N_CH, 1))
        ph = rng.uniform(0.0, 2.0 * math.pi, (n, ae.N_CH, 1))
        x += rng.uniform(1.0, 5.0, (n, ae.N_CH, 1)) * np.sin(2.0 * math.pi * f * t + ph)
    x += rng.normal(0.0, 2.0, x.shape) + rng.normal(0.0, 4.0, (n, ae.N_CH, 1)) * t
    x += rng.uniform(-40.0, 40.0, (1, ae.N_CH, 1)) + rng.normal(0.0, 3.0, (n, ae.N_CH, 1))
    topo_erp = np.array([0.3, 0.5, 0.8, 0.5, 1.0, 0.6, 0.5, 0.6])
    topo_errp = np.array([1.0, 0.45, 1.0, 0.45, 0.5, 0.15, 0.1, 0.15])
    erp = rng.uniform(1.5, 4.0, n)[:, None] * _gauss(t, 0.30 + rng.uniform(-0.03, 0.03, n), 0.06)
    x += topo_erp[None, :, None] * erp[:, None, :]
    lat = rng.uniform(-0.03, 0.03, n)
    errp = rng.uniform(0.7, 1.3, n)[:, None] * (-8.0 * _gauss(t, 0.25 + lat, 0.035)
                                                + 10.0 * _gauss(t, 0.40 + lat, 0.06))
    x[err] += topo_errp[None, :, None] * errp[err][:, None, :]
    return x.astype(np.float32), err


def set_fixture_weights(model, rng: np.random.Generator) -> None:
    """Pesos deterministas con el RNG de numpy: kernels Glorot uniforme, bias N(0, 0.05) y
    BatchNorm NO trivial (gamma U(0.7, 1.3), beta N(0, 0.2), media móvil N(0, 0.3), varianza
    móvil U(0.5, 2)) para que la conversión tenga que plegarla de verdad."""
    for layer in model.layers:
        kind = type(layer).__name__
        ws = layer.get_weights()
        if not ws:
            continue
        if kind == "BatchNormalization":
            g, b, mm, mv = ws
            new = [rng.uniform(0.7, 1.3, g.shape), rng.normal(0.0, 0.2, b.shape),
                   rng.normal(0.0, 0.3, mm.shape), rng.uniform(0.5, 2.0, mv.shape)]
        else:
            new = []
            for w in ws:
                if w.ndim >= 2:
                    receptive = int(np.prod(w.shape[:-2])) if w.ndim > 2 else 1
                    fan_in, fan_out = receptive * w.shape[-2], receptive * w.shape[-1]
                    lim = math.sqrt(6.0 / (fan_in + fan_out))
                    new.append(rng.uniform(-lim, lim, w.shape))
                else:
                    new.append(rng.normal(0.0, 0.05, w.shape))
        layer.set_weights([np.asarray(v, dtype=np.float32) for v in new])


_SHARED_OBJECT_ID = re.compile(rb'"shared_object_id": (\d+)')


def normalize_keras_zip(raw: bytes) -> bytes:
    """.keras determinista: fecha fija en las entradas del zip y en metadata.json["date_saved"]
    (Keras guarda la hora actual) y los "shared_object_id" de config.json renumerados 1, 2, ... por
    orden de aparición (Keras escribe el id() de Python del objeto compartido, p.ej. la política de
    dtype: cambia en cada proceso; renumerar conserva qué capas lo comparten). model.weights.h5 ya
    es determinista."""
    src = zipfile.ZipFile(io.BytesIO(raw))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "metadata.json":
                meta = json.loads(data.decode("utf-8"))
                if "date_saved" in meta:
                    meta["date_saved"] = FIXTURE_DATE
                data = json.dumps(meta).encode("utf-8")
            elif item.filename == "config.json":
                ids: Dict[bytes, bytes] = {}
                data = _SHARED_OBJECT_ID.sub(
                    lambda m: b'"shared_object_id": ' + ids.setdefault(m.group(1), str(len(ids) + 1).encode("ascii")),
                    data)
            info = zipfile.ZipInfo(item.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = item.compress_type
            info.external_attr = item.external_attr
            dst.writestr(info, data)
    return buf.getvalue()


def keras_model_bytes(model) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ae_fixture_") as d:
        path = os.path.join(d, "model.keras")  # Keras exige la extensión .keras
        with captured_output():
            model.save(path)
        raw = Path(path).read_bytes()
    return normalize_keras_zip(raw)


def build_fixture(arch: str, seed: int) -> Dict[str, bytes]:
    """Fixture determinista SIN entrenar: modelo de errp_ae con pesos del RNG de numpy, datos
    sintéticos para rep/val y calibración por defecto (norm = estadísticas de las épocas
    "de entrenamiento"; umbrales = percentiles del score float de las correctas de val).
    Devuelve {nombre: bytes} de model.keras, config.json, rep.npz y val.npz."""
    if arch not in ae.ARCHS:
        raise ExportError(f"--arch {arch!r} no válida (opciones: {', '.join(ae.ARCHS)})")
    _, keras = import_tf()
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng([seed, 7001])
    with captured_output():
        model = ae.build_model(arch)
    set_fixture_weights(model, rng)

    win_tr, _ = fixture_windows(rng, FIXTURE_N_TRAIN, 0)
    win_val, is_err = fixture_windows(rng, FIXTURE_N_VAL_CORRECT, FIXTURE_N_VAL_ERROR)
    ep_tr = ae.preprocess_window(win_tr)
    ep_val = ae.preprocess_window(win_val)
    mean, std = ae.norm_stats(ep_tr)
    z_tr = ae.normalize(ep_tr, mean, std)
    z_val = ae.normalize(ep_val, mean, std)
    s_val = float_scores(model, z_val)
    pcts = ae.DEFAULT_PERCENTILES
    thresholds = ae.thresholds_from_scores(s_val[~is_err], pcts)
    m_val = np.repeat(mean[None, :], z_val.shape[0], axis=0)
    s_std = np.repeat(std[None, :], z_val.shape[0], axis=0)
    config = {
        "_comentario": ("Fixture de export_to_c.py --make-fixture: pesos ALEATORIOS (RNG de numpy) y datos "
                        "SINTETICOS, no es un modelo entrenado. Solo para probar la tuberia de exportacion."),
        "arch": arch,
        "channels": list(ae.CHANNELS),
        "fs_hz": ae.FS_HZ,
        "decim": ae.DECIM,
        "epoch_ms": list(ae.EPOCH_MS),
        "baseline_ms": list(ae.BASELINE_MS),
        "norm": {"mean": [json_f32(v) for v in mean], "std": [json_f32(v) for v in std]},
        "percentiles": [int(p) if float(p).is_integer() else float(p) for p in pcts],
        "thresholds": {ae.percentile_key(p): json_f32(t) for p, t in zip(pcts, thresholds)},
        "score": "mse",
        "placeholder": True,
        "fixture": {"seed": seed, "n_rep": int(z_tr.shape[0]), "n_val_correct": FIXTURE_N_VAL_CORRECT,
                    "n_val_error": FIXTURE_N_VAL_ERROR},
    }
    return {
        "model.keras": keras_model_bytes(model),
        "config.json": (json.dumps(config, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
        "rep.npz": npz_bytes({"z": z_tr}),
        "val.npz": npz_bytes({"z": z_val, "is_error": is_err, "epochs": ep_val, "mean": m_val, "std": s_std,
                              "score": s_val}),
    }


def make_fixture(out_dir: Path, arch: str = "errp_conv", seed: int = 0) -> Dict[str, Path]:
    """Escribe DIR/model.keras, config.json ("placeholder": true), rep.npz y val.npz (deterministas)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ExportError(f"--make-fixture: {out_dir} existe y no es un directorio")
    files = build_fixture(arch, seed)
    paths = {name: out_dir / name for name in FIXTURE_FILES}
    write_files_atomic([(paths[name], files[name]) for name in FIXTURE_FILES])
    return paths


def _display_script() -> str:
    script = Path(__file__).resolve()
    try:
        rel = os.path.relpath(script)
    except ValueError:  # otra unidad en Windows
        return str(script)
    return str(script) if rel.startswith("..") else rel


def print_fixture_summary(paths: Dict[str, Path], arch: str, seed: int) -> None:
    cfg = json.loads(paths["config.json"].read_text(encoding="ascii"))
    print(f"Fixture generado ({arch}, semilla {seed}; pesos ALEATORIOS + datos sintéticos, placeholder):")
    for name in FIXTURE_FILES:
        print(f"  {name:<12}: {paths[name]}")
    print("  umbrales    : " + "  ".join(f"{k}={v}" for k, v in cfg["thresholds"].items()))
    print("Siguiente paso (exportar):")
    d = paths["model.keras"].parent
    print(f'  python "{_display_script()}" --model "{d / "model.keras"}" --config "{d / "config.json"}" '
          f'--rep "{d / "rep.npz"}" --val "{d / "val.npz"}"')


# ---- Autotest ----

class SelftestFailure(Exception):
    pass


C_FLOAT_LITERAL = re.compile(r"-?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)f")
LITERAL_CASES = (
    (1.0, "1.0f"), (-0.0, "-0.0f"), (0.1, "0.100000001f"), (1e-45, "1.40129846e-45f"),
    (1.17549435e-38, "1.17549435e-38f"), (3.40282347e38, "3.40282347e+38f"),
    (123456789.0, "123456792.0f"), (90.0, "90.0f"),
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SelftestFailure(msg)


def _expect_error(fn: Callable[[], object], fragment: str, what: str) -> str:
    try:
        fn()
    except ExportError as e:
        _check(fragment.lower() in str(e).lower(), f"{what}: mensaje inesperado: {e}")
        return str(e)
    raise SelftestFailure(f"{what}: se esperaba un ExportError y no lo hubo")


def parse_c_defines(text: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^#define (\w+)[ \t]+(.*?)[ \t]*(?://[^\n]*)?$", text, re.M)}


def parse_c_arrays(text: str) -> Dict[str, Tuple[str, List[str], List[str]]]:
    """{nombre: (tipo, dimensiones, tokens)} de cada 'static const <tipo> NOMBRE[..] = { ... };'."""
    code = re.sub(r"//[^\n]*", "", text)
    pattern = (r"static const (float|int|int8_t|unsigned char|unsigned int) (\w+)((?:\[\w*\])*)"
               r"(?: __attribute__\(\(aligned\(16\)\)\))? = \{(.*?)\};")
    return {m.group(2): (m.group(1), re.findall(r"\[(\w*)\]", m.group(3)),
                         [t for t in re.split(r"[\s,{}]+", m.group(4)) if t])
            for m in re.finditer(pattern, code, re.S)}


def _floats(tokens: Sequence[str], what: str) -> np.ndarray:
    for tok in tokens:
        _check(C_FLOAT_LITERAL.fullmatch(tok) is not None, f"{what}: literal C no válido {tok!r}")
    return np.array([float(t[:-1]) for t in tokens], dtype=np.float32)


def _ints(tokens: Sequence[str], what: str, lo: int, hi: int) -> np.ndarray:
    for tok in tokens:
        _check(re.fullmatch(r"-?\d+", tok) is not None and lo <= int(tok) <= hi, f"{what}: entero no válido {tok!r}")
    return np.array([int(t) for t in tokens], dtype=np.int64)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    b = np.ascontiguousarray(b, dtype=np.float32).ravel()
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def _selftest_literals() -> str:
    for value, expected in LITERAL_CASES:
        got = c_float(value)
        _check(got == expected, f"c_float({value!r}) = {got!r}; se esperaba {expected!r}")
    for bad in (math.nan, math.inf, -math.inf, 1e39):
        _expect_error(lambda: c_float(bad), "no finito", f"c_float({bad!r})")
    _check(c_int(-3) == "(-3)" and c_int(0) == "0" and c_int(127) == "127", "c_int")
    _check(corr_text(0.9999999) == "0.999999" and corr_text(0.98) == "0.980000" and corr_text(1.0) == "1.000000",
           "corr_text debe truncar a 6 decimales")
    _check(_ops_text(["A", "B", "B", "B", "A"]) == "A, B x3, A", "_ops_text")
    rng = np.random.default_rng(1234)
    bits = rng.integers(0, 2 ** 32, size=12000, dtype=np.uint64).astype(np.uint32)
    values = bits.view(np.float32)
    values = values[np.isfinite(values)][:10000]
    _check(values.size == 10000, "no hay 10000 float32 finitos aleatorios")
    for v in values:
        lit = c_float(v)
        _check(C_FLOAT_LITERAL.fullmatch(lit) is not None, f"literal no válido para {v!r}: {lit!r}")
        _check(np.float32(float(lit[:-1])).view(np.uint32) == v.view(np.uint32), f"ida y vuelta inexacta: {v!r}")
    return f"literales float: {len(LITERAL_CASES)} casos límite + 10000 float32 aleatorios (bits exactos)"


def _selftest_emulator() -> str:
    """Casos a mano de la aritmética de TFLM (C++ de quantization_util.cc, common.cc y gemmlowp)."""
    q30 = 1 << 30
    for m, expected in ((0.5, (q30, 0)), (1.0, (q30, 1)), (0.75, (1610612736, 0)), (0.0, (0, 0)),
                        (2.0 ** -40, (0, 0)), (1.0 - 2.0 ** -33, (q30, 1)), (0.5 + 2.0 ** -32, (q30 + 1, 0)),
                        (0.3, (1288490189, -1))):
        got = quantize_multiplier(m)
        _check(got == expected, f"quantize_multiplier({m!r}) = {got}; se esperaba {expected}")
    near1 = (1 << 31) - 1
    for x, mult, shift, expected in ((100, q30, 0, 50), (3, q30, 0, 2), (-3, q30, 0, -1), (3, q30, -1, 1),
                                     (3, near1, -1, 2), (-3, near1, -1, -2), (1, near1, -1, 1), (-1, near1, -1, -1),
                                     (INT32_MIN, INT32_MIN, 0, INT32_MAX), (12345, 0, 0, 0)):
        got = int(mul_by_quantized_multiplier(x, mult, shift))
        _check(got == expected, f"MultiplyByQuantizedMultiplier({x}, {mult}, {shift}) = {got}; se esperaba {expected}")
    _expect_error(lambda: mul_by_quantized_multiplier(q30, q30, 2), "desborda", "x << shift fuera de int32")
    vec = mul_by_quantized_multiplier(np.array([[3, -3], [100, -100]]), np.array([q30, near1]), np.array([0, -1]))
    _check(vec.tolist() == [[2, -2], [50, -50]], f"MultiplyByQuantizedMultiplier vectorizado: {vec.tolist()}")
    return ("emulación TFLM: QuantizeMultiplier (8 casos: mitad lejos de cero, 2^31 -> shift + 1, shift < -31) y "
            "MultiplyByQuantizedMultiplier con doble redondeo (11 casos: empates, saturación, desbordamiento)")


def _check_lines(text: str, what: str, limit: int = 120) -> None:
    longest = max((len(line) for line in text.splitlines()), default=0)
    _check(longest <= limit, f"{what}: línea de {longest} caracteres (> {limit})")


def _selftest_weights_text(r: ExportResult) -> Dict[str, object]:
    """Comprueba el header de pesos contra el resultado y devuelve lo parseado."""
    text = r.weights_text
    _check(text.isascii() and "\r" not in text, "header de pesos: no es ASCII con '\\n'")
    _check_lines(text, "header de pesos")
    _check("\\" not in text and ":/" not in text and "Users" not in text, "header de pesos: contiene rutas")
    _check(text.startswith("#ifndef AUTOENCODER_WEIGHTS_H\n#define AUTOENCODER_WEIGHTS_H\n// GENERADO"),
           "header de pesos: cabecera inesperada")
    _check(text.endswith("#endif // AUTOENCODER_WEIGHTS_H\n"), "header de pesos: final inesperado")
    d = parse_c_defines(text)
    expected = {
        "AE_MODEL_ID": f'"{r.model_id}"', "AE_MODEL_ARCH": f'"{r.cfg.arch}"',
        "AE_WEIGHTS_PLACEHOLDER": "1" if r.cfg.placeholder else "0", "INPUT_DIM": "320", "OUTPUT_DIM": "320",
        "AE_MODEL_N_CH": "8", "AE_MODEL_N_T": "40", "AE_IN_SCALE": c_float(r.info.in_scale),
        "AE_IN_ZERO_POINT": c_int(r.info.in_zero_point), "AE_OUT_SCALE": c_float(r.info.out_scale),
        "AE_OUT_ZERO_POINT": c_int(r.info.out_zero_point), "AE_TENSOR_ARENA_BYTES": str(r.arena_bytes),
        "AE_INT8_SCORE_CORR": corr_text(r.ver.corr) + "f", "AE_MODEL_N_PARAMS": str(r.n_params),
        "AE_MODEL_N_MACS": str(r.n_macs),
    }
    for k in range(3):
        expected[f"THRESHOLD_LEVEL_{k + 1}"] = c_float(r.cfg.thresholds[k])
        expected[f"AE_CALIB_PCT_{k + 1}"] = c_float(r.cfg.percentiles[k])
    for macro, value in expected.items():
        _check(d.get(macro) == value, f"{macro} = {d.get(macro)!r}; se esperaba {value!r}")
    _check(re.fullmatch(r'"[0-9a-f]{16}"', d["AE_MODEL_ID"]) is not None, "AE_MODEL_ID no son 16 hex")
    for macro, scale in (("AE_IN_SCALE", r.info.in_scales[0]), ("AE_OUT_SCALE", r.info.out_scales[0])):
        _check(float(np.float32(float(d[macro][:-1]))) == scale, f"{macro}: ida y vuelta float32 inexacta")
    arrays = parse_c_arrays(text)
    _check(set(arrays) == {"MEAN_VECTOR", "STD_VECTOR", "AE_MODEL_TFLITE"}, f"arrays: {sorted(arrays)}")
    mean = _floats(arrays["MEAN_VECTOR"][2], "MEAN_VECTOR")
    std = _floats(arrays["STD_VECTOR"][2], "STD_VECTOR")
    _check(_same_bits(mean, r.cfg.mean) and _same_bits(std, r.cfg.std), "MEAN_VECTOR / STD_VECTOR distintos")
    toks = arrays["AE_MODEL_TFLITE"][2]
    _check(all(re.fullmatch(r"0x[0-9a-f]{2}", t) for t in toks), "AE_MODEL_TFLITE: bytes mal escritos")
    blob = bytes(int(t, 16) for t in toks)
    _check(blob == r.tflite, "AE_MODEL_TFLITE != bytes del .tflite")
    m = re.search(r"^static const unsigned int AE_MODEL_TFLITE_LEN = (\d+);$", text, re.M)
    _check(m is not None and int(m.group(1)) == len(r.tflite), "AE_MODEL_TFLITE_LEN incorrecto")
    _check("__attribute__((aligned(16)))" in text, "AE_MODEL_TFLITE sin alineación a 16")
    pcts = tuple(float(d[f"AE_CALIB_PCT_{k}"][:-1]) for k in (1, 2, 3))
    thr = tuple(float(np.float32(float(d[f"THRESHOLD_LEVEL_{k}"][:-1]))) for k in (1, 2, 3))
    cfg2 = ExportConfig(r.cfg.arch, mean, std, pcts, thr, r.cfg.placeholder)  # type: ignore[arg-type]
    _check(compute_model_id(blob, cfg2) == r.model_id, "AE_MODEL_ID no se reproduce desde el header")
    return {"tflite": blob, "mean": mean, "std": std, "thresholds": thr, "percentiles": pcts}


def _selftest_vectors_text(r: ExportResult, runner) -> List[int]:
    """Comprueba cada golden contra errp_ae y el runtime de q_out (recalculados desde el texto)."""
    text = r.vectors_text or ""
    g = r.goldens
    _check(g is not None and text.isascii() and "\r" not in text, "header de vectores: no es ASCII con '\\n'")
    assert g is not None
    _check("\\" not in text and ":/" not in text, "header de vectores: contiene rutas")
    _check_lines(text, "header de vectores")
    d = parse_c_defines(text)
    n, w, nc, ns = len(g.vectors), g.windows.shape[0], g.cal_epochs.shape[0], g.cal_scores.size
    _check(d.get("AE_TV_MODEL_ID") == f'"{r.model_id}"', "AE_TV_MODEL_ID != AE_MODEL_ID")
    _check(d.get("AE_TV_COUNT") == str(n) and d.get("AE_TV_N_WIN") == str(w)
           and d.get("AE_TV_CAL_N_EPOCHS") == str(nc) and d.get("AE_TV_CAL_N_SCORES") == str(ns), "cuentas AE_TV_*")
    _check(ns >= ae.CALIB_MIN_SCORES and w >= 1 and nc >= 1, "goldens de calibración/preprocesado insuficientes")
    a = parse_c_arrays(text)
    sizes = {"AE_TV_WIN": w * ae.N_CH * ae.WIN, "AE_TV_WIN_EPOCH": w * ae.N_IN, "AE_TV_EPOCH": n * ae.N_IN,
             "AE_TV_Z": n * ae.N_IN, "AE_TV_Q_IN": n * ae.N_IN, "AE_TV_Q_OUT": n * ae.N_IN, "AE_TV_ZHAT": n * ae.N_IN,
             "AE_TV_SCORE": n, "AE_TV_LEVEL": n, "AE_TV_NEAR_THRESHOLD": n, "AE_TV_CAL_EPOCHS": nc * ae.N_IN,
             "AE_TV_CAL_MEAN": ae.N_CH, "AE_TV_CAL_STD": ae.N_CH, "AE_TV_CAL_SCORES": ns, "AE_TV_CAL_T": 3}
    _check(set(a) == set(sizes), f"arrays de vectores: {sorted(a)}")
    for name, size in sizes.items():
        _check(len(a[name][2]) == size, f"{name}: {len(a[name][2])} valores, se esperaban {size}")
    _check(a["AE_TV_Q_IN"][0] == "int8_t" and a["AE_TV_Q_OUT"][0] == "int8_t", "q_in/q_out deben ser int8_t")
    win = _floats(a["AE_TV_WIN"][2], "AE_TV_WIN").reshape(w, ae.N_CH, ae.WIN)
    _check(_same_bits(_floats(a["AE_TV_WIN_EPOCH"][2], "AE_TV_WIN_EPOCH"), ae.preprocess_window(win)),
           "AE_TV_WIN_EPOCH != preprocess_window(AE_TV_WIN)")
    ep = _floats(a["AE_TV_EPOCH"][2], "AE_TV_EPOCH").reshape(n, ae.N_CH, ae.N_T)
    z = _floats(a["AE_TV_Z"][2], "AE_TV_Z").reshape(n, ae.N_IN)
    q_in = _ints(a["AE_TV_Q_IN"][2], "AE_TV_Q_IN", -128, 127).reshape(n, ae.N_IN)
    q_out = _ints(a["AE_TV_Q_OUT"][2], "AE_TV_Q_OUT", -128, 127).reshape(n, ae.N_IN)
    zh = _floats(a["AE_TV_ZHAT"][2], "AE_TV_ZHAT").reshape(n, ae.N_IN)
    sc = _floats(a["AE_TV_SCORE"][2], "AE_TV_SCORE")
    lv = _ints(a["AE_TV_LEVEL"][2], "AE_TV_LEVEL", 0, 3)
    near = _ints(a["AE_TV_NEAR_THRESHOLD"][2], "AE_TV_NEAR_THRESHOLD", 0, 1)
    info, cfg = r.info, r.cfg
    for k in range(n):
        _check(_same_bits(ae.normalize(ep[k], cfg.mean, cfg.std).reshape(-1), z[k]), f"vector {k}: Z != normalize")
        qk = ae.quantize(z[k], info.in_scale, info.in_zero_point)
        _check(np.array_equal(qk.astype(np.int64), q_in[k]), f"vector {k}: Q_IN != quantize(Z)")
        _check(np.array_equal(runner.run(qk).reshape(-1).astype(np.int64), q_out[k]),
               f"vector {k}: Q_OUT != {r.golden_runtime}(Q_IN)")
        zhk = ae.dequantize(q_out[k].astype(np.int8), info.out_scale, info.out_zero_point)
        _check(_same_bits(zhk, zh[k]), f"vector {k}: ZHAT != dequantize(Q_OUT)")
        sk = ae.score(z[k], zh[k])
        _check(_same_bits(sk, sc[k]), f"vector {k}: SCORE != score(Z, ZHAT)")
        _check(lv[k] == ae.level(sc[k], *cfg.thresholds), f"vector {k}: LEVEL incoherente")
        _check(near[k] == int(any(abs(float(sc[k]) - t) <= NEAR_REL * t for t in cfg.thresholds)),
               f"vector {k}: NEAR incoherente")
    _check(not np.any(z[0]) and _same_bits(ep[0], np.repeat(cfg.mean[:, None], ae.N_T, axis=1)),
           "el vector 0 debe ser la época = MEAN_VECTOR (z = 0)")
    _check(np.any(q_in == 127) and np.any(q_in == -128), "ningún golden satura q_in")
    counts = [int(np.sum(lv == k)) for k in range(4)]
    t1, t2, t3 = cfg.thresholds
    nonempty = (True, t1 < t2, t2 < t3, True)
    for k in range(4):
        _check(not nonempty[k] or int(np.sum((lv == k) & (near == 0))) >= 1,
               f"banda de nivel {k} sin vector dorado (lejos de los umbrales): {counts}")
    cal = _floats(a["AE_TV_CAL_EPOCHS"][2], "AE_TV_CAL_EPOCHS").reshape(nc, ae.N_CH, ae.N_T)
    m_ref, s_ref = ae.norm_stats(cal)
    _check(_same_bits(_floats(a["AE_TV_CAL_MEAN"][2], "AE_TV_CAL_MEAN"), m_ref)
           and _same_bits(_floats(a["AE_TV_CAL_STD"][2], "AE_TV_CAL_STD"), s_ref), "AE_TV_CAL_MEAN/STD != norm_stats")
    t_ref = ae.thresholds_from_scores(_floats(a["AE_TV_CAL_SCORES"][2], "AE_TV_CAL_SCORES"), cfg.percentiles)
    _check(_same_bits(_floats(a["AE_TV_CAL_T"][2], "AE_TV_CAL_T"), np.array(t_ref, dtype=np.float32)),
           "AE_TV_CAL_T != thresholds_from_scores(AE_TV_CAL_SCORES)")
    return counts


def _tiny_model(keras, kind: str):
    """Modelos de prueba con E/S (8, 40, 1) y una capa sin op permitida (selftest)."""
    L = keras.layers
    inp = keras.Input(shape=ae.INPUT_SHAPE)
    if kind == "elu":
        x = L.Flatten()(inp)
        x = L.Dense(16, activation="elu")(x)
    else:  # Conv2DTranspose -> TRANSPOSE_CONV
        x = L.Conv2DTranspose(2, (1, 3), strides=(1, 2), padding="same")(inp)
        x = L.Flatten()(x)
    x = L.Dense(ae.N_IN)(x)
    return keras.Model(inp, L.Reshape(ae.INPUT_SHAPE)(x))


def _snapshot(directory: Path) -> Dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir())} if Path(directory).is_dir() else {}


def _selftest_body(tmp: Path, warns: List[str]) -> List[str]:
    steps = [_selftest_literals(), _selftest_emulator()]
    tf, keras = import_tf()

    # Fixture determinista
    fx = make_fixture(tmp / "fx", "errp_conv", 0)
    fx2 = make_fixture(tmp / "fx2", "errp_conv", 0)
    for name in FIXTURE_FILES:
        _check(fx[name].read_bytes() == fx2[name].read_bytes(), f"fixture no determinista: {name}")
    fx3 = make_fixture(tmp / "fx3", "errp_conv", 1)
    _check(fx["model.keras"].read_bytes() != fx3["model.keras"].read_bytes(), "fixture: la semilla no influye")
    cfg_fx = json.loads(fx["config.json"].read_text(encoding="ascii"))
    _check(cfg_fx["placeholder"] is True and cfg_fx["arch"] == "errp_conv", "fixture: config.json")
    # Dentro de un proceso los id() coinciden: comprobar que se renumeraron (bytes iguales entre procesos)
    with zipfile.ZipFile(fx["model.keras"]) as zf:
        shared = sorted({int(v) for v in _SHARED_OBJECT_ID.findall(zf.read("config.json"))})
    _check(shared == list(range(1, len(shared) + 1)), f"fixture: shared_object_id sin normalizar: {shared[:3]}")
    steps.append("fixture errp_conv determinista (misma semilla -> mismos bytes de los 4 archivos; otra semilla "
                 f"-> otro modelo; {len(shared)} shared_object_id renumerados: mismos bytes entre procesos)")

    def export(fxp: Dict[str, Path], sub: str, **kw) -> ExportResult:
        base = {"out_path": tmp / sub / "w.h", "tv_path": tmp / sub / "tv.h", "tflite_out": tmp / sub / "m.tflite"}
        base.update(kw)
        return run_export(fxp["model.keras"], fxp["config.json"], fxp["rep.npz"], fxp["val.npz"], **base)

    def golden_runner(r: ExportResult):
        """El runtime con que se generaron los q_out dorados de r (para re-verificarlos)."""
        return TflmReference(r.tflite) if r.golden_runtime == "tflm" else RefInterpreter(tf, r.tflite, r.info)

    runtime_names = {"tflm": "la emulación de TFLM", "builtin_ref": "BUILTIN_REF"}

    # Exportación + headers + determinismo
    r1 = export(fx, "a")
    _check(r1.golden_runtime == DEFAULT_GOLDEN_RUNTIME, f"runtime por defecto {r1.golden_runtime}")
    _check(r1.info.ops == ["CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED",
                           "FULLY_CONNECTED", "FULLY_CONNECTED", "RESHAPE"], f"ops errp_conv: {r1.info.ops}")
    _check(r1.ver.corr >= DEFAULT_MIN_CORR, f"correlación del fixture {r1.ver.corr}")
    _check((tmp / "a" / "m.tflite").read_bytes() == r1.tflite, "--tflite-out != AE_MODEL_TFLITE")
    _check((tmp / "a" / "w.h").read_text(encoding="ascii") == r1.weights_text, "el header escrito difiere")
    _selftest_weights_text(r1)
    counts = _selftest_vectors_text(r1, golden_runner(r1))
    steps.append(f"exportación errp_conv: ops {_ops_text(r1.info.ops)}; correlación {corr_text(r1.ver.corr)}, "
                 f"acuerdo de nivel {100 * r1.ver.agree:.1f} %; headers coherentes (AE_MODEL_TFLITE == .tflite, "
                 f"AE_MODEL_ID reproducible, líneas <= 120) y {len(r1.goldens.vectors) if r1.goldens else 0} "
                 f"goldens verificados contra errp_ae + {runtime_names[r1.golden_runtime]} (niveles {counts})")
    r2 = export(fx, "b")
    for name in ("w.h", "tv.h", "m.tflite"):
        _check((tmp / "a" / name).read_bytes() == (tmp / "b" / name).read_bytes(),
               f"exportación no determinista: {name}")
    _check(r2.model_id == r1.model_id, "AE_MODEL_ID distinto en la segunda exportación")
    steps.append(f"determinismo: dos exportaciones -> mismos bytes (w.h, tv.h, .tflite; AE_MODEL_ID {r1.model_id})")

    # Goldens con el otro runtime: mismo modelo y AE_MODEL_ID, q_out de ese runtime
    other = "builtin_ref" if DEFAULT_GOLDEN_RUNTIME == "tflm" else "tflm"
    ro = export(fx, "t", golden_runtime=other)
    _check(ro.tflite == r1.tflite and ro.model_id == r1.model_id, "--golden-runtime cambió el modelo o AE_MODEL_ID")
    counts_o = _selftest_vectors_text(ro, golden_runner(ro))
    _check(ro.xcheck_val is not None and ro.xcheck_tv is not None and r1.xcheck_tv is not None,
           "falta el contraste BUILTIN_REF / emulación")
    assert ro.xcheck_val is not None and ro.xcheck_tv is not None
    _check(ro.xcheck_val.max_lsb <= XCHECK_MAX_LSB, f"emulación lejos de BUILTIN_REF: {ro.xcheck_val.text()}")
    markers = {"tflm": "emulacion entera", "builtin_ref": "OpResolverType.BUILTIN_REF"}
    _check(f"--golden-runtime {other}" in ro.weights_text and "--golden-runtime" not in r1.weights_text
           and markers[other] in (ro.vectors_text or "") and markers[r1.golden_runtime] in (r1.vectors_text or ""),
           "el runtime de los goldens no consta en los headers")
    steps.append(f"--golden-runtime {other}: mismo .tflite y AE_MODEL_ID; goldens verificados contra "
                 f"{runtime_names[other]} (niveles {counts_o}); BUILTIN_REF vs emulación en val.npz: "
                 f"{ro.xcheck_val.text()}")

    # Fallback denso
    fxd = make_fixture(tmp / "fxd", "dense", 0)
    rd = export(fxd, "d", arena_bytes=12345)
    _check(rd.info.ops == ["RESHAPE"] + ["FULLY_CONNECTED"] * 4 + ["RESHAPE"], f"ops dense: {rd.info.ops}")
    _selftest_weights_text(rd)
    counts_d = _selftest_vectors_text(rd, golden_runner(rd))
    _check("#define AE_TENSOR_ARENA_BYTES 12345" in rd.weights_text and "--arena-bytes 12345" in rd.weights_text,
           "--arena-bytes no llega al header")
    steps.append(f"exportación dense: ops {_ops_text(rd.info.ops)}; correlación {corr_text(rd.ver.corr)}; goldens "
                 f"verificados (niveles {counts_d}); --arena-bytes en el header")

    # Fallos: nada escrito, salidas existentes intactas, sin temporales
    guard = tmp / "guard"
    guard.mkdir()
    for name in ("w.h", "tv.h", "m.tflite"):
        (guard / name).write_bytes(b"previo " + name.encode())
    before = _snapshot(guard)

    def fails(what: str, fragment: str, fxp: Dict[str, Path], **kw) -> str:
        kw.setdefault("out_path", guard / "w.h")
        kw.setdefault("tv_path", guard / "tv.h")
        kw.setdefault("tflite_out", guard / "m.tflite")
        msg = _expect_error(lambda: run_export(fxp["model.keras"], fxp["config.json"], fxp["rep.npz"], fxp["val.npz"],
                                               **kw), fragment, what)
        _check(_snapshot(guard) == before, f"{what}: la exportación fallida tocó las salidas")
        return msg

    bad_dir = tmp / "bad"
    bad_dir.mkdir()
    for kind, op in (("elu", "ELU"), ("convT", "TRANSPOSE_CONV")):
        p = bad_dir / f"{kind}.keras"
        with captured_output():
            _tiny_model(keras, kind).save(str(p))
        msg_op = fails(f"modelo con {op}", "ops no permitidas en el S3", dict(fxd, **{"model.keras": p}))
        _check(op in msg_op.split("(ops del modelo")[0], f"modelo con {op}: el mensaje no nombra la op: {msg_op}")
    msg = fails("--min-corr 0.9999999", "--min-corr", fx, min_corr=0.9999999)
    fails("arquitectura distinta de 'arch'", "no es la arquitectura", dict(fx, **{"config.json": fxd["config.json"]}))
    steps.append("rechazos sin escribir nada: ops ELU y TRANSPOSE_CONV, --min-corr 0.9999999 ("
                 + _brief(msg, 70) + "...), arquitectura distinta de 'arch'")

    # Validadores de config.json / npz / rutas
    base = json.loads(fx["config.json"].read_text(encoding="ascii"))

    def cfg_with(**changes: object) -> Dict[str, object]:
        out = json.loads(json.dumps(base))
        for k, v in changes.items():
            if v is None:
                del out[k]
            else:
                out[k] = v
        return out

    std0 = dict(base["norm"], std=[1.0] * 7 + [0.0])
    for raw, frag, what in (
            (cfg_with(arch="lstm"), "'arch'", "arch desconocida"),
            (cfg_with(norm=std0), "'norm.std' debe ser > 0", "std 0"),
            (cfg_with(norm=dict(base["norm"], mean=[0.0] * 7)), "7 valores", "mean de 7"),
            (cfg_with(thresholds={"p90": 2.0, "p97": 1.0, "p99": 3.0}), "no monótonos", "umbrales no monótonos"),
            (cfg_with(thresholds={"p90": 1.0, "p97": 2.0}), "'p99'", "falta p99"),
            (cfg_with(thresholds={"p90": 0.0, "p97": 2.0, "p99": 3.0}), "> 0", "T1 = 0"),
            (cfg_with(percentiles=[90, 97, 99.5]), "'p99.5'", "percentiles sin su umbral"),
            (cfg_with(percentiles=[97, 90, 99]), "percentiles no válidos", "percentiles decrecientes"),
            (cfg_with(channels=list(reversed(ae.CHANNELS))), "'channels'", "canales en otro orden"),
            (cfg_with(fs_hz=500), "'fs_hz'", "fs_hz 500"),
            (cfg_with(epoch_ms=[-100, 800]), "'epoch_ms'", "epoch_ms"),
            (cfg_with(score="mae"), "'score'", "score mae"),
            (cfg_with(placeholder="no"), "'placeholder'", "placeholder no bool"),
            (cfg_with(placeholder=None), "faltan claves", "falta placeholder"),
            ([1, 2], "objeto JSON", "config no es un objeto")):
        _expect_error(lambda: parse_config(raw, "cfg"), frag, what)
    tuned = parse_config(cfg_with(percentiles=[90, 97, 99.5], thresholds={"p90": 1.0, "p97": 2.0, "p99.5": 3.0}), "cfg")
    _check(tuned.percentiles == (90.0, 97.0, 99.5) and tuned.thresholds == (1.0, 2.0, 3.0), "percentiles TUNE")
    z_ok = np.zeros((30, ae.N_CH, ae.N_T), np.float32)
    z_nan = z_ok.copy()
    z_nan[4, 2, 3] = np.nan
    for arrays, fn, frag, what in (
            ({"x": z_ok}, parse_rep, "falta 'z'", "rep sin z"),
            ({"z": z_ok[:5]}, parse_rep, "al menos", "rep con 5 épocas"),
            ({"z": z_ok[:, :, :20]}, parse_rep, "forma", "rep con forma mala"),
            ({"z": z_nan}, parse_rep, "no finitos", "rep con NaN"),
            ({"z": z_ok}, parse_val, "is_error", "val sin is_error"),
            ({"z": z_ok, "is_error": np.full(30, 2)}, parse_val, "0/1", "is_error no binario")):
        _expect_error(lambda: fn(arrays, "npz"), frag, what)
    _check(parse_val({"z": z_ok[..., None], "is_error": np.zeros(30, np.int64)}, "npz").is_error.dtype == np.bool_,
           "val con z [N, 8, 40, 1] e is_error 0/1")
    bad_json = bad_dir / "bad.json"
    bad_json.write_text("{ no es json", encoding="ascii")
    for what, frag, kw, fxp in (
            ("config inválido", "no es un JSON válido", {}, dict(fx, **{"config.json": bad_json})),
            ("modelo inexistente", "no existe --model", {}, dict(fx, **{"model.keras": bad_dir / "no.keras"})),
            ("val no npz", "no es un .npz", {}, dict(fx, **{"val.npz": bad_json})),
            ("--out == --test-vectors", "mismo archivo", {"tv_path": guard / "w.h"}, fx),
            ("--out directorio", "directorio", {"out_path": guard}, fx),
            ("--tflite-out sobre la entrada", "sobrescribiría", {"tflite_out": fx["rep.npz"]}, fx),
            ("--min-corr 2", "--min-corr", {"min_corr": 2.0}, fx),
            ("--arena-bytes 10", "--arena-bytes", {"arena_bytes": 10}, fx),
            ("--golden-runtime desconocido", "--golden-runtime", {"golden_runtime": "gpu"}, fx)):
        fails(what, frag, fxp, **kw)
    leftovers = [p.name for p in tmp.rglob("*.tmp")]
    _check(not leftovers, f"temporales sin borrar: {leftovers}")
    steps.append("validadores: config.json (15 casos, incl. percentiles TUNE), rep/val.npz (6 casos), rutas y "
                 "opciones (9 casos); salidas previas intactas y sin temporales")
    del warns[:]
    return steps


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ae_export_selftest_"))
    try:
        with captured_warnings() as warns, contextlib.redirect_stdout(io.StringIO()):
            steps = _selftest_body(tmp, warns)
    except (SelftestFailure, ExportError) as e:
        print(f"selftest FALLÓ: {e}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for step in steps:
        print(f"  ok  {step}")
    print("selftest OK")
    return 0


# ---- CLI ----

def _int_range(lo: int, hi: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"entero no válido: {text!r}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"debe estar entre {lo} y {hi}")
        return value
    return parse


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"número no válido: {text!r}") from None
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError("debe estar en [0, 1]")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="model.keras (errp_ae, entrada/salida (8, 40, 1))")
    ap.add_argument("--config", type=Path, help="config.json del entrenamiento (formato arriba)")
    ap.add_argument("--rep", type=Path, help="rep.npz: 'z' [n, 8, 40] épocas correctas normalizadas (PTQ int8)")
    ap.add_argument("--val", type=Path, help="val.npz: 'z' [M, 8, 40] + 'is_error' [M] (correlación y goldens)")
    ap.add_argument("--out", type=Path,
                    help="header de pesos (def. firmware/detector_s3/include/autoencoder_weights.h)")
    tv = ap.add_mutually_exclusive_group()
    tv.add_argument("--test-vectors", type=Path,
                    help="header de vectores dorados (def. firmware/detector_s3/test/autoencoder_test_vectors.h); "
                         "--out y --test-vectors se cambian juntos")
    tv.add_argument("--no-test-vectors", action="store_true",
                    help="no generar los vectores dorados (quedarán desfasados respecto a los pesos)")
    ap.add_argument("--tflite-out", type=Path, help="escribir también el .tflite (mismos bytes que AE_MODEL_TFLITE)")
    ap.add_argument("--min-corr", type=_unit_float, default=DEFAULT_MIN_CORR,
                    help=f"correlación float-int8 mínima del score en val.npz (def. {DEFAULT_MIN_CORR:g}, CLAUDE.md)")
    ap.add_argument("--arena-bytes", type=_int_range(ARENA_MIN, ARENA_MAX), default=None,
                    help=f"AE_TENSOR_ARENA_BYTES: arena TFLM medida + margen (def.: estimación {DEFAULT_ARENA_BYTES}, "
                         f"{ARENA_NOTE})")
    ap.add_argument("--golden-runtime", choices=GOLDEN_RUNTIMES, default=None,
                    help="q_out de los vectores dorados: tflm (def.: emulación entera de los kernels de referencia de "
                         "TFLite Micro, bit a bit con TFLM real) o builtin_ref (tf.lite BUILTIN_REF, CONTRACT v3; en "
                         "TF 2.21 difiere 1 LSB en empates de FULLY_CONNECTED y el nivel B de test_c_engine.c falla)")
    ap.add_argument("--seed", type=_int_range(0, 2 ** 31 - 1), default=0,
                    help="semilla de la conversión, los vectores dorados y el fixture (def. 0)")
    ap.add_argument("--make-fixture", type=Path, metavar="DIR",
                    help="escribir en DIR un modelo ALEATORIO + config/rep/val sintéticos (placeholder)")
    ap.add_argument("--arch", choices=ae.ARCHS, default=None, help="con --make-fixture: arquitectura (def. errp_conv)")
    ap.add_argument("--selftest", action="store_true", help="autotest en un directorio temporal (no toca el repo)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # UTF-8 también al redirigir a un pipe en Windows
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    export_opts = (args.model, args.config, args.rep, args.val)
    export_mode = any(v is not None for v in export_opts)
    modes = int(args.selftest) + int(args.make_fixture is not None) + int(export_mode)
    if modes == 0:
        ap.error("indicar --model/--config/--rep/--val (o --make-fixture DIR, o --selftest)")
    if modes > 1:
        ap.error("elegir un solo modo: exportar (--model ...), --make-fixture DIR o --selftest")
    if export_mode and any(v is None for v in export_opts):
        ap.error("--model, --config, --rep y --val van juntos")
    if args.arch is not None and args.make_fixture is None:
        ap.error("--arch solo vale con --make-fixture (al exportar, la arquitectura sale de config.json)")
    only_export = {"--out": args.out, "--test-vectors": args.test_vectors, "--tflite-out": args.tflite_out,
                   "--arena-bytes": args.arena_bytes, "--golden-runtime": args.golden_runtime}
    if not export_mode:
        used = [k for k, v in only_export.items() if v is not None] + (["--no-test-vectors"] if args.no_test_vectors
                                                                       else [])
        if args.min_corr != DEFAULT_MIN_CORR:
            used.append("--min-corr")
        if used:
            ap.error(f"{', '.join(used)} solo {'vale' if len(used) == 1 else 'valen'} al exportar (--model ...)")
    out_path = args.out if args.out is not None else DEFAULT_WEIGHTS
    tv_path = None if args.no_test_vectors else (
        args.test_vectors if args.test_vectors is not None else DEFAULT_TEST_VECTORS)
    if export_mode and tv_path is not None and (
            _same_file(out_path, DEFAULT_WEIGHTS) != _same_file(tv_path, DEFAULT_TEST_VECTORS)):
        # Uno en el repo y el otro fuera: los headers del repo quedarían desfasados entre sí
        ap.error("--out y --test-vectors se cambian juntos (o usar --no-test-vectors): si solo uno sale de su "
                 "ruta por defecto, los pesos y los vectores del repo quedan desfasados")
    try:
        if args.selftest:
            return selftest()
        if args.make_fixture is not None:
            arch = args.arch or "errp_conv"
            paths = make_fixture(args.make_fixture, arch, args.seed)
            print_fixture_summary(paths, arch, args.seed)
            return 0
        result = run_export(args.model, args.config, args.rep, args.val, out_path, tv_path, args.tflite_out,
                            args.min_corr, args.seed, args.arena_bytes,
                            args.golden_runtime or DEFAULT_GOLDEN_RUNTIME)
        print_summary(result)
        return 0
    except ExportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
