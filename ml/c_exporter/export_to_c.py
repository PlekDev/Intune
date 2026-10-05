#!/usr/bin/env python3
"""
Exporta el autoencoder de anomalías (PyTorch) a un header C estático para el
detector ESP32-S3: inferencia nativa en float32, sin TFLite Micro/ESP-DL/ESP-DSP.

Escribe dos archivos generados (versionados en el repo, NO editarlos a mano):
  firmware/detector_s3/include/autoencoder_weights.h    normalización, pesos y umbrales
  firmware/detector_s3/test/autoencoder_test_vectors.h  vectores dorados de test_c_engine.c
Los dos llevan el mismo AE_MODEL_ID: el test en C falla si quedan desfasados.

Modelo esperado: exactamente 4 nn.Linear 40 -> 16 -> 6 -> 16 -> 40 (OUTPUT_DIM ==
INPUT_DIM), cada una seguida como mucho de UNA activación: identity, relu,
leaky_relu, elu, tanh, sigmoid, silu, gelu o gelu_tanh. Dropout se ignora (eval).
Nada de BatchNorm/LayerNorm, skip connections ni normalización dentro de forward().

Cómo guardar el modelo al entrenar (por orden de preferencia):
  torch.save(model.state_dict(), "ae.pt")  # recomendado: solo tensores, carga segura;
                                           # activaciones en config.json o --activations
  torch.jit.script(model).save("ae.pt")    # TorchScript: activaciones introspectadas y
                                           # verificación contra el forward() original
  torch.save(model, "ae.pt")               # pickle completo: EJECUTA código al cargar (solo
                                           # archivos de confianza) y la clase debe ser importable
También vale un checkpoint {"model_state_dict": ..., "epoch": ...} (claves
state_dict, model_state_dict, model, net o autoencoder).

Score: el MISMO que debe usar el entrenamiento para calcular los percentiles:
  z     = (x - mean) / std          # x: las 40 features crudas, en el orden del entrenamiento
  z_hat = model(z)                  # el modelo trabaja en el espacio normalizado
  score = mean((z - z_hat) ** 2)    # MSE por ventana en el espacio NORMALIZADO
  p95, p99, p99.9 = percentiles de score sobre datos NORMALES (p.ej. validación)
En el S3: score >= p99.9 -> 3 (Severe), >= p99 -> 2 (Moderate), >= p95 -> 1 (Mild),
si no 0 (Normal). Score o entrada no finitos -> 3 (fail-safe).

Formato de config.json (ejemplo completo: --make-fixture DIR):
  {
    "mean": [40 números], "std": [40 números > 0],    # p.ej. StandardScaler: mean_, scale_
    "thresholds": {"p95": 1.21, "p99": 1.48, "p99.9": 1.87},
    "activations": ["leaky_relu:0.1", "tanh", "elu:1.0", "identity"],   # opcional
    "score": "mse",                                                    # opcional
    "feature_names": ["f00", ..., "f39"],                              # opcional
    "placeholder": false                                               # opcional
  }
  Los umbrales también se aceptan en el nivel superior o bajo "percentiles", y
  p99.9 puede escribirse "p99.9", "p999" o "p99_9". Las demás claves se ignoran.
  Activaciones, de mayor a menor prioridad: --activations, "activations" del
  config.json, módulos del modelo (solo si el .pt trae la arquitectura).

Antes de escribir nada se comprueba que la red reconstruida en numpy (pesos ya
redondeados a float32 + activaciones resueltas) reproduce el forward de PyTorch.

Uso:
  python ml/c_exporter/export_to_c.py --model ae.pt --config config.json
  python ml/c_exporter/export_to_c.py --model ae.pt --config config.json --activations relu,identity,relu,identity
  python ml/c_exporter/export_to_c.py --model ae.pt --config config.json --out /tmp/w.h --test-vectors /tmp/tv.h
  python ml/c_exporter/export_to_c.py --make-fixture /tmp/ae_fixture   # modelo ALEATORIO de prueba
  python ml/c_exporter/export_to_c.py --selftest                       # autotest, no toca el repo
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pickle
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

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depende del entorno
    sys.stderr.write("ERROR: falta PyTorch en este entorno de Python (pip install torch)\n")
    raise SystemExit(1)

# ---- Rutas y constantes ----

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "firmware" / "detector_s3" / "include" / "autoencoder_weights.h"
DEFAULT_TEST_VECTORS = REPO_ROOT / "firmware" / "detector_s3" / "test" / "autoencoder_test_vectors.h"
SCRIPT_REL = "ml/c_exporter/export_to_c.py"

EXPECTED_DIMS = (40, 16, 6, 16, 40)  # INPUT, LAYER1, LATENT, LAYER3, OUTPUT (AE_N_FEATURES = 40)
DIM_MACROS = ("INPUT_DIM", "LAYER1_DIM", "LATENT_DIM", "LAYER3_DIM", "OUTPUT_DIM")
THRESHOLD_INFO = (("p95", "Mild"), ("p99", "Moderate"), ("p99.9", "Severe"))
THRESHOLD_KEYS = (("p95", ("p95",)), ("p99", ("p99",)), ("p99.9", ("p99.9", "p999", "p99_9")))
CHECKPOINT_KEYS = ("state_dict", "model_state_dict", "model", "net", "autoencoder")

VERIFY_SCALES = (1.0, 3.0, 10.0)  # z ~ N(0, s^2), además de z = 0
VERIFY_PER_SCALE = 85             # 1 + 3 * 85 = 256 vectores de verificación
VERIFY_TOL = 1e-4                 # |numpy - torch| <= 1e-4 * (1 + |torch|)
NEAR_REL = 1e-3                   # |score - Tk| <= 1e-3 * Tk -> AE_TV_NEAR_THRESHOLD = 1
DEFAULT_N_VECTORS = 32
MIN_N_VECTORS = 11                # z = 0, 2 dirigidos por nivel (8), >= 1 N(0,1) y >= 1 N(0,3^2)
MAX_N_VECTORS = 4096
DIRECTED_TRIES = 24               # direcciones aleatorias por vector dirigido
ANCHOR_ITERS = 100                # iteraciones z <- z_hat si x = MEAN ya no es Normal
VALUES_PER_LINE = 8

FIXTURE_MODEL_NAME = "fixture_ae.pt"
FIXTURE_CONFIG_NAME = "fixture_config.json"
FIXTURE_ACTIVATIONS = ("leaky_relu:0.1", "tanh", "elu:1.0", "identity")
FIXTURE_SAMPLES = 20000

# ---- Activaciones: mismos códigos AE_ACT_* que autoencoder_engine.h ----

(ACT_IDENTITY, ACT_RELU, ACT_LEAKY_RELU, ACT_ELU, ACT_TANH,
 ACT_SIGMOID, ACT_SILU, ACT_GELU, ACT_GELU_TANH) = range(9)
ACT_C_NAMES = ("IDENTITY", "RELU", "LEAKY_RELU", "ELU", "TANH", "SIGMOID", "SILU", "GELU", "GELU_TANH")
ACT_TEXT = ("identity", "relu", "leaky_relu", "elu", "tanh", "sigmoid", "silu", "gelu", "gelu_tanh")
ACT_ALIASES = {
    "identity": ACT_IDENTITY, "none": ACT_IDENTITY, "linear": ACT_IDENTITY,
    "relu": ACT_RELU, "leaky_relu": ACT_LEAKY_RELU, "elu": ACT_ELU, "tanh": ACT_TANH,
    "sigmoid": ACT_SIGMOID, "silu": ACT_SILU, "swish": ACT_SILU,
    "gelu": ACT_GELU, "gelu_tanh": ACT_GELU_TANH,
}
ACT_DEFAULT_PARAM = {ACT_LEAKY_RELU: 0.01, ACT_ELU: 1.0}  # negative_slope / alpha por defecto de PyTorch
ACT_HELP = ("identity|none|linear, relu, leaky_relu[:pendiente], elu[:alpha], tanh, sigmoid, "
            "silu|swish, gelu, gelu_tanh")

# Clases de torch.nn reconocidas al recorrer un módulo (por nombre: vale también para TorchScript)
TORCH_LINEAR = {"Linear", "NonDynamicallyQuantizableLinear"}
TORCH_ACTIVATIONS = {"ReLU", "LeakyReLU", "ELU", "Tanh", "Sigmoid", "SiLU", "GELU"}
TORCH_NOOP = {"Identity", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
              "FeatureAlphaDropout"}  # sin efecto en eval()
TORCH_UNSUPPORTED = {"ReLU6", "PReLU", "RReLU", "SELU", "CELU", "GLU", "Hardtanh", "Hardsigmoid",
                     "Hardswish", "Hardshrink", "Softshrink", "Tanhshrink", "LogSigmoid", "Softplus",
                     "Softsign", "Mish", "Threshold", "Softmax", "Softmin", "LogSoftmax", "Softmax2d"}


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


# ---- Utilidades de float32 y texto ----

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


def short_float(value: float) -> str:
    """Texto más corto que vuelve al mismo float32 (comentarios, mensajes y JSON)."""
    return str(to_f32(value))


_COMMENT_UNSAFE = re.compile(r"[^A-Za-z0-9 _\-.,:;()\[\]{}+=<>%#&@!|^~/'\"]")


def ascii_comment(text: str, max_len: int = 48) -> str:
    """Texto seguro dentro de un comentario // de C: ASCII, sin '*' (nada de '/*' ni '*/'),
    '\\' (empalme de línea), '?' (trigrafos) ni saltos de línea. Las tildes se pliegan (ó -> o)."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    safe = _COMMENT_UNSAFE.sub("_", plain).strip()
    return safe[:max_len] or "_"


def fmt_dims(dims: Sequence[int]) -> str:
    return " -> ".join(str(d) for d in dims)


def _brief(message: str, limit: int = 300) -> str:
    """Resume un mensaje de error largo de PyTorch (sin códigos ANSI)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", message)
    m = re.search(r"WeightsUnpickler error:\s*([^\n]*)", text)
    line = m.group(1) if m else next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line[:limit]


def _keys(d: Mapping) -> str:
    return ", ".join(repr(k) for k in list(d.keys())[:30]) or "(ninguna)"


def write_atomic(path: Path, content) -> None:
    """Escritura atómica: temporal en el mismo directorio y os.replace (crea los directorios)."""
    path = Path(path)
    data = content.encode("ascii") if isinstance(content, str) else bytes(content)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    except OSError as e:
        raise ExportError(f"no se puede escribir en {path.parent}: {e}") from None
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
            raise ExportError(f"no se pudo escribir {path}: {e}") from None
        raise


def _same_file(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- Activaciones ----

@dataclass(frozen=True)
class Activation:
    code: int     # AE_ACT_*
    param: float  # ya redondeado a float32; 0.0 si la activación no tiene parámetro

    @property
    def c_name(self) -> str:
        return "AE_ACT_" + ACT_C_NAMES[self.code]

    def text(self) -> str:
        name = ACT_TEXT[self.code]
        return f"{name}:{short_float(self.param)}" if self.code in ACT_DEFAULT_PARAM else name


IDENTITY = Activation(ACT_IDENTITY, 0.0)


def make_activation(code: int, param: Optional[float] = None, where: str = "activación") -> Activation:
    if code not in ACT_DEFAULT_PARAM:
        return Activation(code, 0.0)
    p = to_f32(ACT_DEFAULT_PARAM[code] if param is None else param)
    if not np.isfinite(p):
        raise ExportError(f"{where}: parámetro de {ACT_TEXT[code]} no finito ({param!r})")
    return Activation(code, float(p))


def parse_activation(text: str, where: str) -> Activation:
    """'relu', 'leaky_relu:0.2', 'ELU:0.5'... (sin distinguir mayúsculas)."""
    raw = str(text)
    name, sep, arg = raw.strip().lower().partition(":")
    name = name.strip()
    code = ACT_ALIASES.get(name)
    if code is None:
        raise ExportError(f"{where}: activación desconocida {raw!r}. Válidas: {ACT_HELP}")
    if not sep:
        return make_activation(code, None, where)
    if code not in ACT_DEFAULT_PARAM:
        raise ExportError(f"{where}: {name} no admite parámetro ({raw!r})")
    try:
        value = float(arg)
    except ValueError:
        raise ExportError(f"{where}: parámetro inválido en {raw!r} (se esperaba un número)") from None
    if not math.isfinite(value):
        raise ExportError(f"{where}: parámetro no finito en {raw!r}")
    return make_activation(code, value, where)


def parse_activation_list(items: Sequence[str], where: str) -> List[Activation]:
    if len(items) != 4:
        raise ExportError(f"{where}: se esperaban 4 activaciones (una por capa Linear) y hay "
                          f"{len(items)}: {list(items)}. Ejemplo: relu,identity,relu,identity")
    return [parse_activation(item, f"{where} [capa {k}]") for k, item in enumerate(items, 1)]


def format_activations(acts: Sequence[Activation]) -> str:
    return ", ".join(a.text() for a in acts)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def np_activation(act: Activation, x: np.ndarray) -> np.ndarray:
    """Activación en float64 con la semántica de PyTorch (la misma que implementa el motor C)."""
    c, p = act.code, act.param
    with np.errstate(over="ignore", invalid="ignore"):
        if c == ACT_IDENTITY:
            return x
        if c == ACT_RELU:
            return np.where(x > 0.0, x, 0.0)
        if c == ACT_LEAKY_RELU:
            return np.where(x > 0.0, x, p * x)
        if c == ACT_ELU:
            return np.where(x > 0.0, x, p * np.expm1(np.minimum(x, 0.0)))
        if c == ACT_TANH:
            return np.tanh(x)
        if c == ACT_SIGMOID:
            return 1.0 / (1.0 + np.exp(-x))
        if c == ACT_SILU:
            return x / (1.0 + np.exp(-x))
        if c == ACT_GELU:
            return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
        if c == ACT_GELU_TANH:
            return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    raise ExportError(f"código de activación desconocido: {c}")


def torch_activation(act: Activation) -> Optional[nn.Module]:
    """Módulo de PyTorch equivalente (None para identity)."""
    c, p = act.code, act.param
    if c == ACT_IDENTITY:
        return None
    if c == ACT_RELU:
        return nn.ReLU()
    if c == ACT_LEAKY_RELU:
        return nn.LeakyReLU(negative_slope=p)
    if c == ACT_ELU:
        return nn.ELU(alpha=p)
    if c == ACT_TANH:
        return nn.Tanh()
    if c == ACT_SIGMOID:
        return nn.Sigmoid()
    if c == ACT_SILU:
        return nn.SiLU()
    if c == ACT_GELU:
        return nn.GELU()
    if c == ACT_GELU_TANH:
        return nn.GELU(approximate="tanh")
    raise ExportError(f"código de activación desconocido: {c}")


# ---- Estructuras ----

@dataclass
class Layer:
    name: str
    weight: np.ndarray  # float32 [out][in], layout de nn.Linear
    bias: np.ndarray    # float32 [out]


@dataclass
class LoadedModel:
    kind: str                                   # texto ASCII: state_dict, TorchScript...
    layers: List[Layer]
    module: Optional[nn.Module] = None          # None si el .pt solo trae pesos
    introspected: Optional[List[Activation]] = None


@dataclass
class ExportConfig:
    mean: np.ndarray
    std: np.ndarray
    thresholds: Tuple[float, float, float]      # float32 (p95, p99, p99.9)
    activations: Optional[List[Activation]]
    feature_names: Optional[List[str]]          # ya saneados para comentarios C
    placeholder: bool


@dataclass
class ExportModel:
    dims: Tuple[int, ...]
    mean: np.ndarray
    std: np.ndarray
    layers: List[Layer]
    thresholds: Tuple[float, float, float]
    activations: List[Activation]
    placeholder: bool
    feature_names: Optional[List[str]]


@dataclass
class GoldenVector:
    label: str
    x: np.ndarray     # float32: features crudas
    z: np.ndarray     # float32: (x - mean) / std en float32
    zhat: np.ndarray  # float32: reconstrucción de PyTorch
    score: float      # float32
    level: int
    near: bool


@dataclass
class Provenance:
    model_name: str
    model_sha256: str
    config_name: str
    config_sha256: str
    kind: str
    activation_source: str
    reference: str
    command: str


@dataclass
class ExportResult:
    model_id: str
    model: ExportModel
    kind: str
    activation_source: str
    reference: str
    max_error: float
    weights_text: str
    vectors_text: Optional[str]
    vectors: List[GoldenVector]
    out_path: Path
    tv_path: Optional[Path]


# ---- Carga del .pt ----

def tensor_to_f32(t: object, what: str) -> np.ndarray:
    """Tensor de PyTorch -> numpy float32 contiguo; rechaza NaN/Inf y valores fuera de float32."""
    if not torch.is_tensor(t):
        raise ExportError(f"{what}: se esperaba un tensor y hay {type(t).__name__}")
    if t.is_quantized or not t.is_floating_point():
        raise ExportError(f"{what}: dtype {t.dtype} no soportado (se esperaba un tensor float)")
    try:
        a64 = t.detach().to(device="cpu", dtype=torch.float64).numpy()
    except Exception as e:  # tensores meta, parámetros sin inicializar...
        raise ExportError(f"{what}: no se pudo leer el tensor ({type(e).__name__}: {e})") from None
    bad = np.flatnonzero(~np.isfinite(a64))
    if bad.size:
        raise ExportError(f"{what}: contiene NaN/Inf ({bad.size} valores, el primero en el "
                          f"índice plano {bad[0]})")
    with np.errstate(over="ignore"):
        a32 = a64.astype(np.float32)
    bad = np.flatnonzero(~np.isfinite(a32))
    if bad.size:
        raise ExportError(f"{what}: valor fuera del rango de float32 ({a64.flat[bad[0]]!r})")
    return np.ascontiguousarray(a32)


def is_torchscript_archive(path: Path) -> bool:
    """Los archivos de torch.jit.save son zip con 'constants.pkl' (torch.save no lo escribe)."""
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            return any(name.endswith("constants.pkl") for name in zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def _torch_load(path: Path, weights_only: bool) -> object:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=weights_only)
    except TypeError as e:
        if "unexpected keyword argument 'weights_only'" not in str(e):
            raise
    # PyTorch < 1.13: no existe weights_only y torch.load siempre usa pickle completo
    if weights_only:
        warn("esta versión de PyTorch no tiene torch.load(weights_only=...): se carga con pickle "
             "completo, que ejecuta código del archivo; usar solo con archivos de confianza")
    return torch.load(str(path), map_location="cpu")


def load_checkpoint(path: Path) -> object:
    """torch.load seguro (weights_only=True); pickle completo solo si el archivo trae clases."""
    try:
        return _torch_load(path, weights_only=True)
    except pickle.UnpicklingError as e:
        message = str(e)
        found = re.search(r"Unsupported global: GLOBAL ([\w.]+)|Unsupported class ([\w.]+)", message,
                          re.IGNORECASE)
        if found is None:
            raise ExportError(f"no se pudo leer {path.name} con torch.load: {_brief(message)}") from None
        needed = found.group(1) or found.group(2)
    except Exception as e:
        raise ExportError(f"no se pudo leer {path.name} con torch.load (¿es un .pt de PyTorch?): "
                          f"{type(e).__name__}: {_brief(str(e))}") from None
    warn(f"{path.name} contiene objetos de Python ({needed}), seguramente un nn.Module guardado con "
         "torch.save(model, ...): se carga con pickle completo (weights_only=False), que EJECUTA "
         "código del archivo. Hacerlo solo con archivos de confianza; lo recomendado es "
         "torch.save(model.state_dict(), ...) o TorchScript")
    try:
        return _torch_load(path, weights_only=False)
    except (AttributeError, ImportError) as e:  # ImportError incluye ModuleNotFoundError
        raise ExportError(
            f"{path.name} guarda un nn.Module completo cuya clase no se puede importar aquí "
            f"({type(e).__name__}: {e}). Opciones: (1) guardar solo los pesos con "
            "torch.save(model.state_dict(), 'ae.pt') e indicar las activaciones (config.json o "
            "--activations); (2) exportar TorchScript: torch.jit.script(model).save('ae.pt'); "
            "(3) añadir al PYTHONPATH el módulo que define la clase (si se guardó desde un script "
            "ejecutado directamente, la clase vive en __main__ y no es importable)") from None
    except Exception as e:
        raise ExportError(f"no se pudo cargar {path.name} con pickle completo: "
                          f"{type(e).__name__}: {_brief(str(e))}") from None


def module_class(m: nn.Module) -> str:
    """Nombre de la clase de torch.nn que implementa m (TorchScript: original_name)."""
    if isinstance(m, torch.jit.ScriptModule):
        return str(getattr(m, "original_name", type(m).__name__))
    for cls in type(m).__mro__:
        if cls.__module__.startswith("torch.nn.modules") and cls is not nn.Module:
            return cls.__name__
    return type(m).__name__


def activation_from_module(m: nn.Module, cls: str, label: str) -> Activation:
    where = f"módulo '{label}'"
    if cls == "ReLU":
        return make_activation(ACT_RELU)
    if cls == "LeakyReLU":
        return make_activation(ACT_LEAKY_RELU, float(getattr(m, "negative_slope", 0.01)), where)
    if cls == "ELU":
        return make_activation(ACT_ELU, float(getattr(m, "alpha", 1.0)), where)
    if cls == "Tanh":
        return make_activation(ACT_TANH)
    if cls == "Sigmoid":
        return make_activation(ACT_SIGMOID)
    if cls == "SiLU":
        return make_activation(ACT_SILU)
    approximate = str(getattr(m, "approximate", "none"))  # GELU
    if approximate == "none":
        return make_activation(ACT_GELU)
    if approximate == "tanh":
        return make_activation(ACT_GELU_TANH)
    raise ExportError(f"{where}: GELU(approximate={approximate!r}) no soportado")


def model_from_module(module: nn.Module, kind: str) -> LoadedModel:
    """Capas Linear en orden de registro (named_modules) y la activación que sigue a cada una."""
    layers: List[Layer] = []
    acts: List[Optional[Tuple[Activation, str]]] = []
    linear_names: List[str] = []
    for name, m in module.named_modules():
        label = name or "(raiz)"
        if any(name.startswith(prefix + ".") for prefix in linear_names):
            continue  # internos de una Linear (p.ej. parametrizaciones de weight_norm)
        cls = module_class(m)
        if cls in TORCH_LINEAR:
            weight = tensor_to_f32(m.weight, f"{label}.weight")
            bias_t = getattr(m, "bias", None)
            bias = (tensor_to_f32(bias_t, f"{label}.bias") if bias_t is not None
                    else np.zeros(weight.shape[0], np.float32))  # bias=False -> ceros
            if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                raise ExportError(f"capa '{label}': formas {weight.shape} / {bias.shape} no válidas")
            layers.append(Layer(label, weight, bias))
            acts.append(None)
            linear_names.append(name)
            continue
        own = ([n for n, _ in m.named_parameters(recurse=False)]
               + [n for n, _ in m.named_buffers(recurse=False)])
        if own:
            raise ExportError(
                f"el módulo '{label}' ({cls}) tiene parámetros/buffers ({', '.join(own)}) que el "
                "motor C no soporta: solo 4 nn.Linear con una activación cada una (sin BatchNorm, "
                "LayerNorm, escalas aprendidas...). Si el modelo normaliza dentro de forward(), "
                "quitar esa normalización y poner mean/std en config.json")
        if any(True for _ in m.children()):
            continue  # contenedor: Sequential, la clase del modelo...
        if cls in TORCH_NOOP:
            continue
        if cls in TORCH_ACTIVATIONS:
            act = activation_from_module(m, cls, label)
            if not layers:
                raise ExportError(f"activación '{label}' ({cls}) antes de la primera capa Linear: "
                                  "no soportado (el motor entra directamente a W1 con z)")
            if acts[-1] is not None:
                raise ExportError(f"más de una activación tras la capa Linear '{layers[-1].name}' "
                                  f"('{acts[-1][1]}' y '{label}'): el motor C aplica una por capa")
            acts[-1] = (act, label)
            continue
        if cls in TORCH_UNSUPPORTED:
            raise ExportError(f"activación '{label}' ({cls}) no soportada por el motor C. "
                              f"Soportadas: {ACT_HELP}")
        warn(f"módulo '{label}' ({cls}) desconocido y sin parámetros: se ignora; la verificación "
             "numérica contra forward() decidirá si eso es correcto")
    introspected = [a[0] if a is not None else IDENTITY for a in acts]
    return LoadedModel(kind, layers, module=module, introspected=introspected)


def layers_from_state_dict(sd: Mapping) -> List[Layer]:
    """Pesos 2-D '*.weight' en orden de inserción con su '*.bias' (sin bias -> ceros)."""
    layers: List[Layer] = []
    used = set()
    for key, t in sd.items():
        if not isinstance(key, str):
            raise ExportError(f"state_dict con una clave no textual: {key!r}")
        if (key == "weight" or key.endswith(".weight")) and t.dim() == 2:
            prefix = key[: -len("weight")]
            weight = tensor_to_f32(t, key)
            bias_key = prefix + "bias"
            if bias_key in sd:
                bias_t = sd[bias_key]
                if bias_t.dim() != 1 or bias_t.shape[0] != weight.shape[0]:
                    raise ExportError(f"{bias_key}: forma {tuple(bias_t.shape)} incompatible con "
                                      f"{key} {tuple(weight.shape)}")
                bias = tensor_to_f32(bias_t, bias_key)
                used.add(bias_key)
            else:
                bias = np.zeros(weight.shape[0], np.float32)
            used.add(key)
            layers.append(Layer(prefix[:-1] or "(raiz)", weight, bias))
    extra = [k for k in sd.keys() if k not in used]
    if extra:
        shown = ", ".join(f"{k} {tuple(sd[k].shape)}" for k in extra[:8])
        more = f" y {len(extra) - 8} más" if len(extra) > 8 else ""
        raise ExportError(f"el state_dict contiene tensores que no son de capas nn.Linear: {shown}{more}. "
                          "El motor C solo admite 4 capas Linear (sin BatchNorm/LayerNorm/...)")
    return layers


def model_from_object(obj: object, kind: str = "", depth: int = 0) -> LoadedModel:
    """Desenvuelve lo que devuelve torch.load: state_dict, checkpoint, nn.Module o TorchScript."""
    if isinstance(obj, torch.jit.ScriptModule):
        return model_from_module(obj, kind + "TorchScript")
    if isinstance(obj, nn.Module):
        return model_from_module(obj, kind + "modulo completo (pickle)")
    if isinstance(obj, Mapping):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return LoadedModel(kind + "state_dict", layers_from_state_dict(obj))
        if depth < 3:
            for key in CHECKPOINT_KEYS:
                if key in obj:
                    return model_from_object(obj[key], f"{kind}checkpoint['{key}'] -> ", depth + 1)
        raise ExportError(f"el .pt contiene un diccionario sin pesos reconocibles (claves: {_keys(obj)}). "
                          "Se esperaba un state_dict (nombre -> tensor), un checkpoint con alguna de "
                          f"las claves {', '.join(CHECKPOINT_KEYS)}, un nn.Module o TorchScript")
    raise ExportError(f"el .pt contiene un objeto {type(obj).__name__}: se esperaba un state_dict, "
                      "un checkpoint, un nn.Module o TorchScript")


def load_model(path: Path) -> LoadedModel:
    if not path.is_file():
        raise ExportError(f"no existe el modelo: {path}")
    if is_torchscript_archive(path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # torch.jit está deprecado en 2.x
                module = torch.jit.load(str(path), map_location="cpu")
        except Exception as e:
            raise ExportError(f"{path.name} parece TorchScript pero torch.jit.load falló: "
                              f"{type(e).__name__}: {_brief(str(e))}") from None
        return model_from_module(module, "TorchScript")
    return model_from_object(load_checkpoint(path))


def validate_layers(layers: List[Layer]) -> Tuple[int, ...]:
    """4 capas, cadena de formas coherente y OUTPUT_DIM == INPUT_DIM. Devuelve las 5 dims."""
    names = ", ".join(f"{l.name} {l.weight.shape[1]}->{l.weight.shape[0]}" for l in layers)
    if len(layers) != 4:
        raise ExportError(f"se esperaban exactamente 4 capas nn.Linear ({fmt_dims(EXPECTED_DIMS)}) y "
                          f"el modelo tiene {len(layers)}: {names or 'ninguna'}")
    for k in range(1, 4):
        prev, cur = layers[k - 1], layers[k]
        if cur.weight.shape[1] != prev.weight.shape[0]:
            raise ExportError(f"cadena de dimensiones inconsistente: la capa {k} ({prev.name}) sale con "
                              f"{prev.weight.shape[0]} y la capa {k + 1} ({cur.name}) espera "
                              f"{cur.weight.shape[1]}. Capas: {names}")
    dims = (int(layers[0].weight.shape[1]),) + tuple(int(l.weight.shape[0]) for l in layers)
    if min(dims) <= 0:
        raise ExportError(f"dimensiones no válidas: {fmt_dims(dims)}")
    if dims[4] != dims[0]:
        raise ExportError(f"OUTPUT_DIM ({dims[4]}) debe ser igual a INPUT_DIM ({dims[0]}): el score "
                          "compara z con su reconstrucción z_hat")
    if dims[0] != EXPECTED_DIMS[0]:
        warn(f"INPUT_DIM = {dims[0]} != {EXPECTED_DIMS[0]}: el motor C exige INPUT_DIM == AE_N_FEATURES "
             f"({EXPECTED_DIMS[0]}) y no compilará con este header")
    elif dims != EXPECTED_DIMS:
        warn(f"dimensiones {fmt_dims(dims)} distintas de las previstas {fmt_dims(EXPECTED_DIMS)} "
             "(el motor C usa los LAYERk_DIM del header, así que compila igual)")
    return dims


# ---- config.json ----

def load_config_json(path: Path) -> Tuple[object, bytes]:
    if not path.is_file():
        raise ExportError(f"no existe el config: {path}")
    data = path.read_bytes()
    try:
        return json.loads(data.decode("utf-8-sig")), data
    except (UnicodeDecodeError, ValueError) as e:  # json.JSONDecodeError es un ValueError
        raise ExportError(f"{path.name} no es un JSON válido: {e}") from None


def _short_repr(value: object, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _number(value: object) -> Optional[float]:
    """Número JSON -> float (inf si no cabe); None si no es un número (bool no cuenta)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _config_vector(raw: Mapping, key: str, n: int, where: str) -> np.ndarray:
    if key not in raw:
        raise ExportError(f"{where}: falta '{key}' (lista de {n} números). Claves encontradas: {_keys(raw)}")
    values = raw[key]
    if not isinstance(values, list):
        raise ExportError(f"{where}: '{key}' debe ser una lista de {n} números (es {type(values).__name__})")
    if len(values) != n:
        raise ExportError(f"{where}: '{key}' tiene {len(values)} valores y el modelo tiene INPUT_DIM = {n}")
    out = np.empty(n, np.float32)
    for i, v in enumerate(values):
        f = _number(v)
        if f is None:
            raise ExportError(f"{where}: '{key}'[{i}] no es un número: {_short_repr(v)}")
        f32 = to_f32(f)
        if not np.isfinite(f32):
            raise ExportError(f"{where}: '{key}'[{i}] = {_short_repr(v)} no es finito en float32")
        out[i] = f32
    return out


def _config_thresholds(raw: Mapping, where: str) -> Tuple[float, float, float]:
    container, place = raw, "el nivel superior"
    for key in ("thresholds", "percentiles"):
        if key in raw:
            container, place = raw[key], f"'{key}'"
            break
    if not isinstance(container, Mapping):
        raise ExportError(f'{where}: {place} debe ser un objeto {{"p95": ..., "p99": ..., "p99.9": ...}}')
    values: List[float] = []
    for canon, aliases in THRESHOLD_KEYS:
        present = [(a, container[a]) for a in aliases if a in container]
        if not present:
            raise ExportError(f"{where}: falta el umbral '{canon}' en {place} (claves encontradas: "
                              f'{_keys(container)}). Formato: "thresholds": {{"p95": 1.2, "p99": 1.5, '
                              '"p99.9": 2.0}')
        nums = []
        for alias, v in present:
            f = _number(v)
            if f is None:
                raise ExportError(f"{where}: el umbral '{alias}' no es un número: {_short_repr(v)}")
            f32 = to_f32(f)
            if not (np.isfinite(f32) and f32 > 0):
                raise ExportError(f"{where}: el umbral '{alias}' = {_short_repr(v)} debe ser finito y > 0 "
                                  "(en float32)")
            nums.append(float(f32))
        if any(x != nums[0] for x in nums):
            raise ExportError(f"{where}: valores distintos para {canon}: "
                              + ", ".join(f"{a}={_short_repr(v)}" for a, v in present))
        values.append(nums[0])
    if not values[0] <= values[1] <= values[2]:
        raise ExportError(f"{where}: umbrales no monótonos: p95={short_float(values[0])}, "
                          f"p99={short_float(values[1])}, p99.9={short_float(values[2])} "
                          "(se exige p95 <= p99 <= p99.9)")
    return values[0], values[1], values[2]


def parse_config(raw: object, input_dim: int, where: str) -> ExportConfig:
    """Valida config.json contra INPUT_DIM del modelo."""
    if not isinstance(raw, Mapping):
        raise ExportError(f"{where}: debe ser un objeto JSON {{...}} (es {type(raw).__name__})")
    mean = _config_vector(raw, "mean", input_dim, where)
    std = _config_vector(raw, "std", input_dim, where)
    bad = np.flatnonzero(~(std > 0))
    if bad.size:
        shown = ", ".join(f"[{i}]={short_float(std[i])}" for i in bad[:8])
        raise ExportError(f"{where}: 'std' debe ser > 0 en todas las features: {shown}")
    tiny = np.flatnonzero(std < 1e-6)
    if tiny.size:
        warn(f"{where}: std < 1e-6 en las features {tiny.tolist()[:8]}: z se dispara con un ruido "
             "mínimo (¿feature constante en el entrenamiento?)")
    thresholds = _config_thresholds(raw, where)
    activations = None
    if raw.get("activations") is not None:
        value = raw["activations"]
        if isinstance(value, str):
            value = value.split(",")
        if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ExportError(f"{where}: 'activations' debe ser una lista de 4 textos, p.ej. "
                              '["relu", "identity", "relu", "identity"]')
        activations = parse_activation_list(value, f"{where} 'activations'")
    score = raw.get("score")
    if score is not None and not (isinstance(score, str) and score.strip().lower() == "mse"):
        raise ExportError(f"{where}: 'score' = {score!r} no soportado: el motor C solo calcula 'mse' "
                          "(media de (z - z_hat)^2 en el espacio normalizado)")
    names = None
    if raw.get("feature_names") is not None:
        value = raw["feature_names"]
        if not (isinstance(value, list) and len(value) == input_dim and all(isinstance(v, str) for v in value)):
            raise ExportError(f"{where}: 'feature_names' debe ser una lista de {input_dim} textos")
        names = [ascii_comment(v, 32) for v in value]
    placeholder = raw.get("placeholder", False)
    if placeholder is None:
        placeholder = False
    if not isinstance(placeholder, bool):
        raise ExportError(f"{where}: 'placeholder' debe ser true o false (es {placeholder!r})")
    return ExportConfig(mean, std, thresholds, activations, names, placeholder)


def resolve_activations(cli: Optional[List[Activation]], cfg: Optional[List[Activation]],
                        introspected: Optional[List[Activation]]) -> Tuple[List[Activation], str]:
    """Prioridad: --activations > config.json > módulos del modelo."""
    if cli is not None:
        acts, source = cli, "--activations"
        if cfg is not None and cfg != cli:
            warn(f"--activations ({format_activations(cli)}) sustituye a las del config.json "
                 f"({format_activations(cfg)})")
    elif cfg is not None:
        acts, source = cfg, "config.json"
    elif introspected is not None:
        return introspected, "modulos del modelo"
    else:
        raise ExportError(
            "el .pt es un state_dict (solo pesos): no hay módulos de los que leer las activaciones. "
            "Indicarlas con --activations relu,identity,relu,identity (4 valores, una por capa Linear, "
            f'en orden) o con "activations" en config.json. Válidas: {ACT_HELP}')
    if introspected is not None and acts != introspected:
        warn(f"las activaciones de {source} ({format_activations(acts)}) no coinciden con los módulos "
             f"del modelo ({format_activations(introspected)}); decide la verificación numérica (es "
             "normal si forward() usa activaciones funcionales como F.relu)")
    return acts, source


# ---- Forward de referencia (PyTorch) y verificación ----

def numpy_forward(layers: Sequence[Layer], acts: Sequence[Activation], z: np.ndarray) -> np.ndarray:
    """Forward en float64 con los pesos exportados (float32) y las fórmulas del motor C."""
    h = np.asarray(z, dtype=np.float64)
    for layer, act in zip(layers, acts):
        h = h @ layer.weight.astype(np.float64).T + layer.bias.astype(np.float64)
        h = np_activation(act, h)
    return h


def build_sequential(layers: Sequence[Layer], acts: Sequence[Activation]) -> nn.Sequential:
    """nn.Sequential equivalente a un state_dict + activaciones (pesos ya en float32)."""
    modules: List[nn.Module] = []
    for layer, act in zip(layers, acts):
        lin = nn.Linear(layer.weight.shape[1], layer.weight.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(layer.weight))
            lin.bias.copy_(torch.from_numpy(layer.bias))
        modules.append(lin)
        act_module = torch_activation(act)
        if act_module is not None:
            modules.append(act_module)
    return nn.Sequential(*modules)


def build_reference(loaded: LoadedModel,
                    acts: Sequence[Activation]) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Forward de PyTorch en eval/no_grad: el del módulo original si existe; si no, un
    nn.Sequential reconstruido. Se evalúa en float64 si el modelo lo admite (referencia más
    precisa y reproducible entre máquinas). Devuelve (función z -> z_hat float64, descripción)."""
    if loaded.module is not None:
        module = loaded.module
        desc = f"forward original ({loaded.kind})"
    else:
        module = build_sequential(loaded.layers, acts)
        desc = "nn.Sequential reconstruido desde el state_dict"
        warn("el .pt solo trae pesos (state_dict): la verificación y AE_TV_ZHAT usan un nn.Sequential "
             "reconstruido con las activaciones indicadas, así que la arquitectura original NO se pudo "
             "comprobar (para comprobarla, exportar TorchScript: torch.jit.script(model).save(...))")
    module.eval()
    d_in = loaded.layers[0].weight.shape[1]
    d_out = loaded.layers[-1].weight.shape[0]
    warned_tuple = []

    def run(z: np.ndarray, dtype: torch.dtype) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float64)).to(dtype)
        with torch.no_grad():
            out = module(x)
        if isinstance(out, (tuple, list)):
            if not warned_tuple:
                warn(f"forward() devuelve un {type(out).__name__} de {len(out)} elementos: se usa el "
                     "elemento 0 como reconstrucción")
                warned_tuple.append(True)
            out = out[0] if len(out) else None
        if not torch.is_tensor(out):
            raise ExportError(f"forward() no devuelve un tensor (devuelve {type(out).__name__})")
        if tuple(out.shape) != (z.shape[0], d_out):
            raise ExportError(f"forward() devuelve forma {tuple(out.shape)} con una entrada "
                              f"({z.shape[0]}, {d_in}); se esperaba ({z.shape[0]}, {d_out})")
        return out.detach().to(device="cpu", dtype=torch.float64).numpy()

    probe = np.zeros((2, d_in))
    dtype = torch.float64
    try:
        module.double()
        run(probe, dtype)
    except ExportError:
        raise
    except Exception as e64:
        dtype = torch.float32
        try:
            module.float()
            run(probe, dtype)
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"el forward del modelo falló con una entrada (2, {d_in}): "
                              f"{type(e).__name__}: {_brief(str(e))}") from None
        warn(f"el modelo no admite float64 ({type(e64).__name__}): la referencia de PyTorch se "
             "calcula en float32")

    def reference(z: np.ndarray) -> np.ndarray:
        try:
            return run(z, dtype)
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"el forward del modelo falló: {type(e).__name__}: {_brief(str(e))}") from None

    return reference, desc + (" en float64" if dtype == torch.float64 else " en float32")


def verify_against_torch(layers: Sequence[Layer], acts: Sequence[Activation],
                         reference: Callable[[np.ndarray], np.ndarray], seed: int) -> float:
    """numpy (pesos float32 exportados + activaciones resueltas) vs forward de PyTorch."""
    rng = np.random.default_rng([seed, 1])
    d = layers[0].weight.shape[1]
    groups = [np.zeros((1, d))] + [s * rng.standard_normal((VERIFY_PER_SCALE, d)) for s in VERIFY_SCALES]
    z = np.concatenate(groups).astype(np.float32)
    expected = reference(z)
    if not np.all(np.isfinite(expected)):
        raise ExportError("el forward de PyTorch devolvió NaN/Inf con entradas finitas: modelo no válido")
    got = numpy_forward(layers, acts, z)
    err = np.abs(got - expected)
    tol = VERIFY_TOL * (1.0 + np.abs(expected))
    bad = ~(err <= tol)  # NaN cuenta como fallo
    if bad.any():
        i, j = (int(v) for v in np.argwhere(bad)[0])
        raise ExportError(
            "verificación numérica FALLIDA, no se escribió nada: la red reconstruida con las "
            f"activaciones [{format_activations(acts)}] no reproduce el forward de PyTorch "
            f"({int(bad.sum())} de {bad.size} salidas fuera de tolerancia; vector {i}, salida {j}: "
            f"numpy={got[i, j]:.6g} torch={expected[i, j]:.6g}, |dif|={err[i, j]:.3g} > {tol[i, j]:.3g}). "
            "Revisar --activations (o 'activations' en config.json): si forward() usa activaciones "
            "funcionales (F.relu, torch.tanh...) no hay módulos que inspeccionar y hay que indicarlas "
            "explícitamente. Comprobar también que forward() sea solo Linear + activación en el orden "
            "de registro (sin skip connections, normalización interna, etc.)")
    return float(err.max())


# ---- Modelo exportado ----

def compute_model_id(m: ExportModel) -> str:
    """sha256 de los bytes float32 LE de MEAN, STD, W1, B1..W4, B4, umbrales y (ACT, PARAM) x 4."""
    h = hashlib.sha256()

    def put(values) -> None:
        h.update(np.ascontiguousarray(values, dtype="<f4").tobytes())

    put(m.mean)
    put(m.std)
    for layer in m.layers:
        put(layer.weight)
        put(layer.bias)
    put(np.array(m.thresholds, dtype=np.float64))
    for act in m.activations:
        put(np.array([act.code, act.param], dtype=np.float64))
    return h.hexdigest()[:16]


def level_of(score: float, thresholds: Sequence[float]) -> int:
    """Regla del contrato en float32: no finito -> 3; score == umbral escala."""
    s = to_f32(score)
    if not np.isfinite(s):
        return 3
    t1, t2, t3 = (np.float32(t) for t in thresholds)
    if s >= t3:
        return 3
    if s >= t2:
        return 2
    if s >= t1:
        return 1
    return 0


def near_threshold(score: float, thresholds: Sequence[float]) -> bool:
    return any(abs(score - t) <= NEAR_REL * t for t in thresholds)


def make_golden_vectors(model: ExportModel, reference: Callable[[np.ndarray], np.ndarray],
                        n: int, seed: int) -> List[GoldenVector]:
    """z = 0, dos vectores dirigidos por nivel y el resto z ~ N(0,1) / N(0,3^2)."""
    d = model.dims[0]
    t1, t2, t3 = model.thresholds
    mean64 = model.mean.astype(np.float64)
    std64 = model.std.astype(np.float64)

    def evaluate(z_target: np.ndarray):
        z_target = np.atleast_2d(z_target)
        with np.errstate(over="ignore"):
            x = (mean64 + z_target * std64).astype(np.float32)
        z = (x - model.mean) / model.std  # float32, misma aritmética que el motor
        zhat = reference(z)
        score = np.mean((z.astype(np.float64) - zhat) ** 2, axis=1)
        return x, z, zhat, score

    def vector(label: str, z_target: np.ndarray) -> GoldenVector:
        x, z, zhat, score = evaluate(z_target)
        s32 = float(to_f32(score[0]))
        return GoldenVector(label, x[0], z[0], zhat[0].astype(np.float32), s32,
                            level_of(s32, model.thresholds), near_threshold(s32, model.thresholds))

    origin = vector("z = 0 (x = MEAN_VECTOR)", np.zeros(d))
    s0 = origin.score
    # Punto base de los vectores dirigidos: z = 0, salvo que x = MEAN ya no sea Normal (score >= T1).
    # Entonces se itera la reconstrucción (z <- z_hat), que en un autoencoder suele acercarse a sus
    # datos y bajar el score, y se parte del punto de menor score encontrado.
    base, base_score = np.zeros(d), s0
    if s0 >= t1:
        z_it = np.zeros(d)
        for _ in range(ANCHOR_ITERS):
            _, z_eval, zhat, score = evaluate(z_it)
            if not (np.all(np.isfinite(zhat)) and math.isfinite(float(score[0]))):
                break
            if score[0] < base_score:
                base, base_score = z_eval[0].astype(np.float64), float(score[0])
            z_it = zhat[0]
    rng_dir = np.random.default_rng([seed, 3])

    def directed(level: int, target: float) -> Optional[GoldenVector]:
        """Escala una dirección aleatoria desde el punto base y biseca la escala hasta score ~= target."""
        if not base_score < target:
            return None
        for _ in range(DIRECTED_TRIES):
            direction = rng_dir.standard_normal(d)

            def score_at(s: float) -> float:
                return float(evaluate(base + s * direction)[3][0])

            lo, hi = 0.0, 1.0
            f_hi = score_at(hi)
            while f_hi < target and hi < 1e6:
                lo, hi = hi, 2.0 * hi
                f_hi = score_at(hi)
            if f_hi < target:
                continue  # esta dirección no llega al objetivo
            for _ in range(60):  # invariante: score(lo) < target <= score(hi)
                if hi - lo <= 1e-6 * hi:
                    break
                mid = 0.5 * (lo + hi)
                if score_at(mid) < target:
                    lo = mid
                else:
                    hi = mid
            v = vector(f"nivel {level} dirigido (score objetivo {target:.6g})", base + hi * direction)
            if v.level == level:
                return v
        return None

    # Objetivos por banda [lo, hi): desde max(lo, score del punto base), porque escalar una
    # dirección solo sube el score a partir de ahí. Bandas vacías (umbrales iguales) no llevan vectores.
    plans: List[Tuple[int, List[float]]] = []
    unreachable: List[int] = []
    for level, (lo, hi) in enumerate(((0.0, t1), (t1, t2), (t2, t3), (t3, math.inf))):
        if not lo < hi:
            continue
        if hi <= base_score:
            unreachable.append(level)
            continue
        start = max(lo, base_score)
        if level == 0:
            targets = [start + (hi - start) * f for f in (0.35, 0.7)]
        elif math.isinf(hi):
            targets = [2.0 * start, 4.0 * start]
        else:
            targets = [start * (hi / start) ** f for f in (1.0 / 3.0, 2.0 / 3.0)]  # puntos geométricos
        plans.append((level, targets))
    if unreachable:
        lowest = ("" if base_score == s0 else
                  f" (y {short_float(base_score)} el menor encontrado iterando la reconstrucción)")
        warn(f"el score con z = 0 (x = MEAN) es {short_float(s0)}{lowest}, ya por encima de los niveles "
             f"{unreachable}: no hay vectores dirigidos de esos niveles y test_c_engine fallará en "
             "test_golden_coverage (¿umbrales o mean/std incoherentes con el modelo?)")

    directed_vectors: List[GoldenVector] = []
    for level, targets in plans:
        for target in targets:
            v = directed(level, target)
            if v is None:
                warn(f"no se pudo generar un vector de nivel {level} (score objetivo {target:.6g}); "
                     "se sustituye por uno aleatorio")
            else:
                directed_vectors.append(v)

    rng = np.random.default_rng([seed, 2])
    n_random = n - 1 - len(directed_vectors)
    n_unit = (n_random + 1) // 2
    random_vectors = [vector("z ~ N(0,1)", rng.standard_normal(d)) for _ in range(n_unit)]
    random_vectors += [vector("z ~ N(0,3^2)", 3.0 * rng.standard_normal(d))
                       for _ in range(n_random - n_unit)]
    vectors = [origin] + random_vectors + directed_vectors
    for i, v in enumerate(vectors):
        if not (np.all(np.isfinite(v.x)) and np.all(np.isfinite(v.z)) and np.all(np.isfinite(v.zhat))
                and math.isfinite(v.score)):
            raise ExportError(f"vector de prueba {i} ({v.label}) no finito: modelo numéricamente inestable")
    return vectors


# ---- Generación de los headers ----

def _float_rows(values: np.ndarray, indent: str) -> List[str]:
    lits = [c_float(v) for v in np.asarray(values).ravel()]
    return [indent + ", ".join(lits[i:i + VALUES_PER_LINE]) + ","
            for i in range(0, len(lits), VALUES_PER_LINE)]


def _int_rows(values: Sequence[int], indent: str, per_line: int = 16) -> List[str]:
    texts = [str(int(v)) for v in values]
    return [indent + ", ".join(texts[i:i + per_line]) + "," for i in range(0, len(texts), per_line)]


def _c_vector(decl: str, values: np.ndarray) -> List[str]:
    return [f"{decl} = {{"] + _float_rows(values, "    ") + ["};"]


def _c_matrix(decl: str, rows: np.ndarray, labels: Sequence[str]) -> List[str]:
    lines = [f"{decl} = {{"]
    for label, row in zip(labels, rows):
        lines.append(f"    {{ // {label}")
        lines += _float_rows(row, "        ")
        lines.append("    },")
    lines.append("};")
    return lines


def _feature_table(names: Sequence[str]) -> List[str]:
    width = max(len(n) for n in names)
    cells = [f"[{i:2d}] {n:<{width}}" for i, n in enumerate(names)]
    per_line = max(1, 92 // (len(cells[0]) + 2))
    lines = ["// Features (indice: nombre), en el orden del entrenamiento:"]
    for i in range(0, len(cells), per_line):
        lines.append(("//   " + "  ".join(cells[i:i + per_line])).rstrip())
    return lines


def render_weights_header(m: ExportModel, model_id: str, prov: Provenance) -> str:
    out: List[str] = []
    add = out.append
    add("#ifndef AUTOENCODER_WEIGHTS_H")
    add("#define AUTOENCODER_WEIGHTS_H")
    add("// GENERADO por ml/c_exporter/export_to_c.py - NO EDITAR A MANO.")
    add("//")
    add(f"// Modelo: {prov.model_name} (sha256 {prov.model_sha256})")
    add(f"//   formato del .pt: {prov.kind}")
    add(f"// Config: {prov.config_name} (sha256 {prov.config_sha256})")
    add(f"// Arquitectura: {fmt_dims(m.dims)} (4 capas densas, float32, W[out][in] como nn.Linear)")
    for k, (layer, act) in enumerate(zip(m.layers, m.activations), 1):
        add(f"//   capa {k}: {layer.weight.shape[1]} -> {layer.weight.shape[0]}, {act.text()} "
            f"({ascii_comment(layer.name)})")
    add(f"// Activaciones tomadas de: {prov.activation_source}; verificadas contra PyTorch")
    add(f"//   ({prov.reference}, tolerancia 1e-4 relativa).")
    add("//")
    add("// Score = MSE en el espacio normalizado (los umbrales son percentiles de esta misma")
    add("// magnitud sobre datos normales):")
    add("//   z[i] = (x[i] - MEAN_VECTOR[i]) / STD_VECTOR[i]")
    add("//   h1 = act1(W1 z + B1); h2 = act2(W2 h1 + B2); h3 = act3(W3 h2 + B3); z_hat = act4(W4 h3 + B4)")
    add("//   score = (1 / INPUT_DIM) * sum_i (z[i] - z_hat[i])^2")
    add("// Nivel: score >= THRESHOLD_LEVEL_3 -> 3, >= THRESHOLD_LEVEL_2 -> 2, >= THRESHOLD_LEVEL_1 -> 1,")
    add("// si no 0. Score o entrada no finitos -> 3 (fail-safe).")
    add("//")
    add("// AE_MODEL_ID: primeros 16 hex del sha256 de los bytes float32 little-endian de MEAN_VECTOR,")
    add("// STD_VECTOR, W1, B1, W2, B2, W3, B3, W4, B4, THRESHOLD_LEVEL_1..3 y, para k = 1..4,")
    add("// (LAYERk_ACT, LAYERk_ACT_PARAM).")
    add("//")
    add("// Regenerar desde la raiz del repo (por defecto reescribe tambien")
    add("// firmware/detector_s3/test/autoencoder_test_vectors.h):")
    add(f"//   {prov.command}")
    add("// Solo lo incluye autoencoder_engine.c (arrays static const: una sola copia en flash).")
    add("")
    add("#ifndef AE_ACT_IDENTITY")
    add('#error "Incluir autoencoder_engine.h antes de autoencoder_weights.h"')
    add("#endif")
    add("")
    for macro, d in zip(DIM_MACROS, m.dims):
        add(f"#define {macro} {d}")
    add("")
    for k, (t, (pname, level)) in enumerate(zip(m.thresholds, THRESHOLD_INFO), 1):
        add(f"#define THRESHOLD_LEVEL_{k} {c_float(t)} // {pname} ({level})")
    add("")
    add("// Activacion de cada capa (codigos AE_ACT_* de autoencoder_engine.h). LAYERk_ACT_PARAM:")
    add("// pendiente negativa (LEAKY_RELU) o alpha (ELU); 0.0f si la activacion no tiene parametro.")
    for k, act in enumerate(m.activations, 1):
        add(f"#define LAYER{k}_ACT {act.c_name}")
        add(f"#define LAYER{k}_ACT_PARAM {c_float(act.param)}")
    add("")
    add(f'#define AE_MODEL_ID "{model_id}"')
    add(f"#define AE_WEIGHTS_PLACEHOLDER {1 if m.placeholder else 0} "
        "// 1 = pesos de prueba/aleatorios (fixture), NO un modelo entrenado")
    add("")
    if m.feature_names:
        out.extend(_feature_table(m.feature_names))
    add("// Normalizacion por feature: z = (x - MEAN_VECTOR) / STD_VECTOR")
    out.extend(_c_vector("static const float MEAN_VECTOR[INPUT_DIM]", m.mean))
    add("")
    out.extend(_c_vector("static const float STD_VECTOR[INPUT_DIM]", m.std))
    shapes = (("LAYER1_DIM", "INPUT_DIM"), ("LATENT_DIM", "LAYER1_DIM"),
              ("LAYER3_DIM", "LATENT_DIM"), ("OUTPUT_DIM", "LAYER3_DIM"))
    for k, (layer, act, (rows, cols)) in enumerate(zip(m.layers, m.activations, shapes), 1):
        add("")
        add(f"// Capa {k}: {ascii_comment(layer.name)} ({layer.weight.shape[1]} -> "
            f"{layer.weight.shape[0]}, {act.text()})")
        labels = [f"W{k}[{r}]" for r in range(layer.weight.shape[0])]
        out.extend(_c_matrix(f"static const float W{k}[{rows}][{cols}]", layer.weight, labels))
        out.extend(_c_vector(f"static const float B{k}[{rows}]", layer.bias))
    add("")
    add("#endif // AUTOENCODER_WEIGHTS_H")
    return "\n".join(out) + "\n"


def render_test_vectors(m: ExportModel, model_id: str, vectors: Sequence[GoldenVector],
                        prov: Provenance, seed: int) -> str:
    counts = [sum(1 for v in vectors if v.level == k) for k in range(4)]
    n_near = sum(1 for v in vectors if v.near)
    labels = [f"[{i}] {ascii_comment(v.label, 64)}" for i, v in enumerate(vectors)]
    out: List[str] = []
    add = out.append
    add("#ifndef AUTOENCODER_TEST_VECTORS_H")
    add("#define AUTOENCODER_TEST_VECTORS_H")
    add("// GENERADO por ml/c_exporter/export_to_c.py - NO EDITAR A MANO.")
    add("//")
    add("// Vectores dorados de firmware/detector_s3/test/test_c_engine.c, generados junto con")
    add("// firmware/detector_s3/include/autoencoder_weights.h (mismo AE_MODEL_ID).")
    add(f"// Modelo: {prov.model_name} (sha256 {prov.model_sha256})")
    add(f"// Config: {prov.config_name} (sha256 {prov.config_sha256}); semilla {seed}.")
    add("//   AE_TV_X     features crudas que recibe el motor (ae_infer).")
    add("//   AE_TV_Z     (x - MEAN_VECTOR) / STD_VECTOR en float32, igual que el motor.")
    add("//   AE_TV_ZHAT  reconstruccion de PyTorch sobre AE_TV_Z, redondeada a float32:")
    add(f"//               {prov.reference}.")
    add("//   AE_TV_SCORE mean((Z - ZHAT)^2) en float64, redondeado a float32.")
    add("//   AE_TV_LEVEL nivel con los umbrales float32 (score == umbral escala).")
    add("//   AE_TV_NEAR_THRESHOLD 1 si |score - Tk| <= 1e-3 * Tk para algun k: el test no exige")
    add("//               el nivel exacto de ese vector.")
    add("// Cobertura: x = MEAN_VECTOR (z = 0); z ~ N(0,1); z ~ N(0,3^2); 2 vectores dirigidos por")
    add("// nivel (escala de una direccion aleatoria biseccionada hasta un score objetivo dentro de")
    add("// la banda: a 1/3 y 2/3 de ella, o 2x y 4x su inicio en el nivel 3).")
    add(f"// Vectores por nivel 0/1/2/3: {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}; "
        f"cerca de un umbral: {n_near}.")
    add("")
    add(f'#define AE_TV_MODEL_ID "{model_id}"')
    add(f"#define AE_TV_COUNT {len(vectors)}")
    add(f"#define AE_TV_DIM {m.dims[0]}")
    add("")
    out.extend(_c_matrix("static const float AE_TV_X[AE_TV_COUNT][AE_TV_DIM]",
                         np.stack([v.x for v in vectors]), labels))
    add("")
    out.extend(_c_matrix("static const float AE_TV_Z[AE_TV_COUNT][AE_TV_DIM]",
                         np.stack([v.z for v in vectors]), [f"[{i}]" for i in range(len(vectors))]))
    add("")
    out.extend(_c_matrix("static const float AE_TV_ZHAT[AE_TV_COUNT][AE_TV_DIM]",
                         np.stack([v.zhat for v in vectors]), [f"[{i}]" for i in range(len(vectors))]))
    add("")
    out.extend(_c_vector("static const float AE_TV_SCORE[AE_TV_COUNT]",
                         np.array([v.score for v in vectors], dtype=np.float32)))
    add("")
    add("static const int AE_TV_LEVEL[AE_TV_COUNT] = {")
    out.extend(_int_rows([v.level for v in vectors], "    "))
    add("};")
    add("")
    add("static const unsigned char AE_TV_NEAR_THRESHOLD[AE_TV_COUNT] = {")
    out.extend(_int_rows([1 if v.near else 0 for v in vectors], "    "))
    add("};")
    add("")
    add("#endif // AUTOENCODER_TEST_VECTORS_H")
    return "\n".join(out) + "\n"


def regen_command(model_name: str, config_name: str, cli_acts: Optional[List[Activation]],
                  n_vectors: int, seed: int) -> str:
    parts = ["python", SCRIPT_REL, "--model", f"<ruta>/{model_name}", "--config", f"<ruta>/{config_name}"]
    if cli_acts is not None:
        parts += ["--activations", ",".join(a.text() for a in cli_acts)]
    if n_vectors != DEFAULT_N_VECTORS:
        parts += ["--n-vectors", str(n_vectors)]
    if seed != 0:
        parts += ["--seed", str(seed)]
    return " ".join(parts)


# ---- Exportación completa ----

def run_export(model_path: Path, config_path: Path, activations: Optional[str] = None,
               out_path: Path = DEFAULT_WEIGHTS, tv_path: Optional[Path] = DEFAULT_TEST_VECTORS,
               n_vectors: int = DEFAULT_N_VECTORS, seed: int = 0) -> ExportResult:
    """Carga, valida, verifica y SOLO al final escribe los headers (atómicamente)."""
    model_path, config_path, out_path = Path(model_path), Path(config_path), Path(out_path)
    tv_path = Path(tv_path) if tv_path is not None else None
    for path in (out_path, tv_path):
        if path is not None and path.is_dir():
            raise ExportError(f"la salida apunta a un directorio: {path}")
    if tv_path is not None and _same_file(out_path, tv_path):
        raise ExportError("--out y --test-vectors apuntan al mismo archivo")
    if not MIN_N_VECTORS <= n_vectors <= MAX_N_VECTORS:
        raise ExportError(f"--n-vectors debe estar entre {MIN_N_VECTORS} y {MAX_N_VECTORS}")
    cli_acts = parse_activation_list(activations.split(","), "--activations") if activations is not None else None

    raw_cfg, cfg_bytes = load_config_json(config_path)
    loaded = load_model(model_path)
    dims = validate_layers(loaded.layers)
    cfg = parse_config(raw_cfg, dims[0], config_path.name)
    acts, act_source = resolve_activations(cli_acts, cfg.activations, loaded.introspected)
    reference, ref_desc = build_reference(loaded, acts)
    max_err = verify_against_torch(loaded.layers, acts, reference, seed)

    model = ExportModel(dims, cfg.mean, cfg.std, loaded.layers, cfg.thresholds, acts,
                        cfg.placeholder, cfg.feature_names)
    model_id = compute_model_id(model)
    model_name, config_name = ascii_comment(model_path.name, 80), ascii_comment(config_path.name, 80)
    prov = Provenance(model_name, _sha256_file(model_path), config_name,
                      hashlib.sha256(cfg_bytes).hexdigest(), ascii_comment(loaded.kind, 80),
                      act_source, ascii_comment(ref_desc, 100),
                      regen_command(model_name, config_name, cli_acts, n_vectors, seed))
    weights_text = render_weights_header(model, model_id, prov)
    vectors: List[GoldenVector] = []
    vectors_text = None
    if tv_path is not None:
        vectors = make_golden_vectors(model, reference, n_vectors, seed)
        vectors_text = render_test_vectors(model, model_id, vectors, prov, seed)
    else:
        warn("--no-test-vectors: no se generan vectores de prueba; un autoencoder_test_vectors.h de "
             "otro modelo hará fallar test_c_engine (AE_TV_MODEL_ID distinto) hasta regenerarlo")

    write_atomic(out_path, weights_text)
    if tv_path is not None and vectors_text is not None:
        write_atomic(tv_path, vectors_text)
    return ExportResult(model_id, model, loaded.kind, act_source, ref_desc, max_err, weights_text,
                        vectors_text, vectors, out_path, tv_path)


def print_summary(r: ExportResult) -> None:
    m = r.model
    t1, t2, t3 = m.thresholds
    print("Exportación completada:")
    print(f"  formato del .pt : {r.kind}")
    print(f"  arquitectura    : {fmt_dims(m.dims)}")
    print(f"  activaciones    : {format_activations(m.activations)} (fuente: {r.activation_source})")
    print(f"  umbrales        : p95={short_float(t1)}  p99={short_float(t2)}  p99.9={short_float(t3)}")
    print(f"  AE_MODEL_ID     : {r.model_id}")
    if m.placeholder:
        print("  placeholder     : SÍ -> AE_WEIGHTS_PLACEHOLDER 1 (pesos de prueba, NO es un modelo entrenado)")
    else:
        print("  placeholder     : no -> AE_WEIGHTS_PLACEHOLDER 0")
    n_verify = 1 + len(VERIFY_SCALES) * VERIFY_PER_SCALE
    print(f"  verificación    : error máx |numpy - torch| = {r.max_error:.3g} en {n_verify} vectores "
          f"(referencia: {r.reference})")
    print(f"  pesos           : {r.out_path}")
    if r.tv_path is not None:
        counts = [sum(1 for v in r.vectors if v.level == k) for k in range(4)]
        n_near = sum(1 for v in r.vectors if v.near)
        print(f"  vectores        : {r.tv_path}")
        print(f"                    {len(r.vectors)} vectores; por nivel 0/1/2/3: "
              f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}; cerca de un umbral: {n_near}")
    else:
        print("  vectores        : no generados (--no-test-vectors)")


# ---- Fixture (modelo aleatorio de prueba) ----

class ReferenceAutoencoder(nn.Module):
    """Autoencoder denso de referencia (encoder/decoder nn.Sequential) con el layout que espera
    el exportador. Para el entrenamiento: torch.save(modelo.state_dict(), "ae.pt")."""

    def __init__(self, dims: Sequence[int] = EXPECTED_DIMS,
                 activations: Sequence[str] = FIXTURE_ACTIVATIONS) -> None:
        super().__init__()
        acts = parse_activation_list(list(activations), "ReferenceAutoencoder")
        blocks: List[List[nn.Module]] = []
        for k in range(4):
            block: List[nn.Module] = [nn.Linear(int(dims[k]), int(dims[k + 1]))]
            act_module = torch_activation(acts[k])
            if act_module is not None:
                block.append(act_module)
            blocks.append(block)
        self.encoder = nn.Sequential(*blocks[0], *blocks[1])
        self.decoder = nn.Sequential(*blocks[2], *blocks[3])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(z))


def build_fixture(seed: int) -> Tuple[ReferenceAutoencoder, Dict[str, object]]:
    """Pesos ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) como el init de torch, pero con numpy
    (independiente del RNG de torch); umbrales = percentiles del score sobre z ~ N(0,1)."""
    rng = np.random.default_rng(seed)
    dims = EXPECTED_DIMS
    acts = parse_activation_list(list(FIXTURE_ACTIVATIONS), "fixture")
    layers = []
    for k in range(4):
        bound = 1.0 / math.sqrt(dims[k])
        weight = rng.uniform(-bound, bound, size=(dims[k + 1], dims[k])).astype(np.float32)
        bias = rng.uniform(-bound, bound, size=dims[k + 1]).astype(np.float32)
        layers.append(Layer(f"capa{k + 1}", weight, bias))
    mean = rng.uniform(1.0, 10.0, size=dims[0]).astype(np.float32)
    std = rng.uniform(0.5, 3.0, size=dims[0]).astype(np.float32)
    z = rng.standard_normal((FIXTURE_SAMPLES, dims[0])).astype(np.float32)
    zhat = numpy_forward(layers, acts, z)
    scores = np.mean((z.astype(np.float64) - zhat) ** 2, axis=1)
    p95, p99, p999 = (float(short_float(v)) for v in np.percentile(scores, [95.0, 99.0, 99.9]))

    module = ReferenceAutoencoder(dims, FIXTURE_ACTIVATIONS)
    linears = [mod for mod in module.modules() if isinstance(mod, nn.Linear)]
    with torch.no_grad():
        for lin, layer in zip(linears, layers):
            lin.weight.copy_(torch.from_numpy(layer.weight))
            lin.bias.copy_(torch.from_numpy(layer.bias))
    module.eval()
    config: Dict[str, object] = {
        "_comentario": ("Fixture de ejemplo de export_to_c.py --make-fixture: pesos ALEATORIOS, no es un "
                        "modelo entrenado. mean/std por feature en el orden del entrenamiento; thresholds = "
                        "percentiles de score = mean(((x - mean) / std - z_hat)^2) sobre datos normales."),
        "mean": [float(short_float(v)) for v in mean],
        "std": [float(short_float(v)) for v in std],
        "thresholds": {"p95": p95, "p99": p99, "p99.9": p999},
        "activations": list(FIXTURE_ACTIVATIONS),
        "score": "mse",
        "feature_names": [f"f{i:02d}" for i in range(dims[0])],
        "placeholder": True,
    }
    return module, config


def make_fixture(out_dir: Path, seed: int) -> Tuple[Path, Path]:
    """Escribe DIR/fixture_ae.pt (state_dict) y DIR/fixture_config.json; deterministas."""
    out_dir = Path(out_dir)
    module, config = build_fixture(seed)
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)  # a memoria: los bytes no dependen del nombre del archivo
    model_path = out_dir / FIXTURE_MODEL_NAME
    config_path = out_dir / FIXTURE_CONFIG_NAME
    write_atomic(model_path, buf.getvalue())
    write_atomic(config_path, json.dumps(config, indent=2, ensure_ascii=True) + "\n")
    return model_path, config_path


def _display_script() -> str:
    script = Path(__file__).resolve()
    try:
        rel = os.path.relpath(script)
    except ValueError:  # otra unidad en Windows
        return str(script)
    return str(script) if rel.startswith("..") else rel


def print_fixture_summary(model_path: Path, config_path: Path, seed: int) -> None:
    thresholds = json.loads(config_path.read_text(encoding="ascii"))["thresholds"]
    print(f"Fixture generado (semilla {seed}; pesos ALEATORIOS, placeholder -> AE_WEIGHTS_PLACEHOLDER 1):")
    print(f"  modelo  : {model_path}")
    print(f"  config  : {config_path}")
    print(f"  umbrales: p95={thresholds['p95']}  p99={thresholds['p99']}  p99.9={thresholds['p99.9']} "
          f"({FIXTURE_SAMPLES} muestras z ~ N(0,1))")
    print("Siguiente paso (exportar):")
    print(f'  python "{_display_script()}" --model "{model_path}" --config "{config_path}"')


# ---- Autotest ----

class SelftestFailure(Exception):
    pass


class _SelftestFunctionalAE(nn.Module):
    """Solo para el selftest: activaciones funcionales en forward() (sin módulos que inspeccionar)."""

    def __init__(self, dims: Sequence[int] = EXPECTED_DIMS) -> None:
        super().__init__()
        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])
        self.l3 = nn.Linear(dims[2], dims[3])
        self.l4 = nn.Linear(dims[3], dims[4])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(z))
        h = torch.relu(self.l2(h))
        h = torch.relu(self.l3(h))
        return self.l4(h)


C_FLOAT_LITERAL = re.compile(r"-?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)f")
LITERAL_CASES = (
    (1.0, "1.0f"), (-0.0, "-0.0f"), (0.1, "0.100000001f"), (1e-45, "1.40129846e-45f"),
    (1.17549435e-38, "1.17549435e-38f"), (3.40282347e38, "3.40282347e+38f"),
    (123456789.0, "123456792.0f"),
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SelftestFailure(msg)


def _expect_error(fn: Callable[[], object], fragment: str, what: str) -> None:
    try:
        fn()
    except ExportError as e:
        _check(fragment.lower() in str(e).lower(), f"{what}: mensaje inesperado: {e}")
        return
    raise SelftestFailure(f"{what}: se esperaba un ExportError y no lo hubo")


def parse_c_arrays(text: str) -> Dict[str, List[str]]:
    """Tokens de cada 'static const <tipo> NOMBRE[..] = { ... };' (sin comentarios //)."""
    code = re.sub(r"//[^\n]*", "", text)
    pattern = r"static const (?:float|int|unsigned char) (\w+)((?:\[\w+\])+) = \{(.*?)\};"
    return {m.group(1): [t for t in re.split(r"[\s,{}]+", m.group(3)) if t]
            for m in re.finditer(pattern, code, re.S)}


def parse_c_defines(text: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^#define (\w+)[ \t]+(.*?)[ \t]*(?://[^\n]*)?$", text, re.M)}


def _floats_from_literals(tokens: Sequence[str], what: str) -> np.ndarray:
    for tok in tokens:
        _check(C_FLOAT_LITERAL.fullmatch(tok) is not None, f"{what}: literal C no válido {tok!r}")
    return np.array([float(t[:-1]) for t in tokens], dtype=np.float32)


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
    rng = np.random.default_rng(1234)
    bits = rng.integers(0, 2 ** 32, size=12000, dtype=np.uint64).astype(np.uint32)
    values = bits.view(np.float32)
    values = values[np.isfinite(values)][:10000]
    _check(values.size == 10000, "no hay 10000 float32 finitos aleatorios")
    values = np.concatenate([values, np.array([v for v, _ in LITERAL_CASES], dtype=np.float32)])
    for v in values:
        lit = c_float(v)
        _check(C_FLOAT_LITERAL.fullmatch(lit) is not None, f"literal no válido para {v!r}: {lit!r}")
        back = np.float32(float(lit[:-1]))
        _check(back.view(np.uint32) == v.view(np.uint32), f"ida y vuelta inexacta: {v!r} -> {lit}")
    return f"literales float: {len(LITERAL_CASES)} casos límite + 10000 float32 aleatorios (bits exactos)"


def _selftest_weights_text(r: ExportResult) -> None:
    text = r.weights_text
    _check(text.isascii() and "\r" not in text, "header de pesos: no es ASCII con '\\n'")
    _check("\\" not in text and ":/" not in text, "header de pesos: contiene rutas")
    _check(text.startswith("#ifndef AUTOENCODER_WEIGHTS_H\n#define AUTOENCODER_WEIGHTS_H\n// GENERADO"),
           "header de pesos: cabecera inesperada")
    _check('#ifndef AE_ACT_IDENTITY\n#error "Incluir autoencoder_engine.h antes de autoencoder_weights.h"'
           in text, "header de pesos: falta el #error de AE_ACT_IDENTITY")
    defines = parse_c_defines(text)
    for macro, d in zip(DIM_MACROS, r.model.dims):
        _check(defines.get(macro) == str(d), f"{macro} = {defines.get(macro)!r}, se esperaba {d}")
    for k, t in enumerate(r.model.thresholds, 1):
        _check(defines.get(f"THRESHOLD_LEVEL_{k}") == c_float(t), f"THRESHOLD_LEVEL_{k} incorrecto")
    for k, act in enumerate(r.model.activations, 1):
        _check(defines.get(f"LAYER{k}_ACT") == act.c_name, f"LAYER{k}_ACT incorrecto")
        _check(defines.get(f"LAYER{k}_ACT_PARAM") == c_float(act.param), f"LAYER{k}_ACT_PARAM incorrecto")
    _check(re.fullmatch(r'"[0-9a-f]{16}"', defines.get("AE_MODEL_ID", "")) is not None
           and defines["AE_MODEL_ID"] == f'"{r.model_id}"', "AE_MODEL_ID incorrecto")
    _check(defines.get("AE_WEIGHTS_PLACEHOLDER") == ("1" if r.model.placeholder else "0"),
           "AE_WEIGHTS_PLACEHOLDER incorrecto")
    arrays = parse_c_arrays(text)
    expected = {"MEAN_VECTOR": r.model.mean, "STD_VECTOR": r.model.std}
    for k, layer in enumerate(r.model.layers, 1):
        expected[f"W{k}"] = layer.weight
        expected[f"B{k}"] = layer.bias
    _check(set(arrays) == set(expected), f"arrays del header: {sorted(arrays)}")
    for name, values in expected.items():
        _check(len(arrays[name]) == values.size,
               f"{name}: {len(arrays[name])} literales, se esperaban {values.size}")
        _check(_same_bits(_floats_from_literals(arrays[name], name), values), f"{name}: valores distintos")


def _selftest_vectors_text(r: ExportResult) -> None:
    text = r.vectors_text or ""
    _check(text.isascii() and "\r" not in text, "header de vectores: no es ASCII con '\\n'")
    defines = parse_c_defines(text)
    n, d = len(r.vectors), r.model.dims[0]
    _check(defines.get("AE_TV_MODEL_ID") == f'"{r.model_id}"', "AE_TV_MODEL_ID != AE_MODEL_ID")
    _check(defines.get("AE_TV_COUNT") == str(n) and n == DEFAULT_N_VECTORS, "AE_TV_COUNT incorrecto")
    _check(defines.get("AE_TV_DIM") == str(d), "AE_TV_DIM incorrecto")
    arrays = parse_c_arrays(text)
    sizes = {"AE_TV_X": n * d, "AE_TV_Z": n * d, "AE_TV_ZHAT": n * d, "AE_TV_SCORE": n,
             "AE_TV_LEVEL": n, "AE_TV_NEAR_THRESHOLD": n}
    _check(set(arrays) == set(sizes), f"arrays de vectores: {sorted(arrays)}")
    for name, size in sizes.items():
        _check(len(arrays[name]) == size, f"{name}: {len(arrays[name])} valores, se esperaban {size}")
    x = _floats_from_literals(arrays["AE_TV_X"], "AE_TV_X").reshape(n, d)
    z = _floats_from_literals(arrays["AE_TV_Z"], "AE_TV_Z").reshape(n, d)
    _floats_from_literals(arrays["AE_TV_ZHAT"], "AE_TV_ZHAT")
    scores = _floats_from_literals(arrays["AE_TV_SCORE"], "AE_TV_SCORE")
    levels = [int(t) for t in arrays["AE_TV_LEVEL"]]
    near = [int(t) for t in arrays["AE_TV_NEAR_THRESHOLD"]]
    _check(_same_bits(x[0], r.model.mean) and not np.any(z[0]), "el vector 0 debe ser x = MEAN (z = 0)")
    _check(_same_bits((x - r.model.mean) / r.model.std, z), "AE_TV_Z != (X - MEAN) / STD en float32")
    for i in range(n):
        _check(levels[i] == level_of(float(scores[i]), r.model.thresholds), f"nivel del vector {i} incoherente")
        _check(near[i] == int(near_threshold(float(scores[i]), r.model.thresholds)), f"NEAR del vector {i}")
    counts = [levels.count(k) for k in range(4)]
    _check(min(counts) >= 2, f"cobertura de niveles insuficiente: {counts}")


def _selftest_body(tmp: Path, warns: List[str]) -> List[str]:
    steps = [_selftest_literals()]

    # Fixture determinista
    m1, c1 = make_fixture(tmp / "fx1", 0)
    m2, c2 = make_fixture(tmp / "fx2", 0)
    _check(c1.read_bytes() == c2.read_bytes() and m1.read_bytes() == m2.read_bytes(),
           "fixture: archivos distintos con la misma semilla")
    _check(m1.read_bytes() != make_fixture(tmp / "fx3", 1)[0].read_bytes(), "fixture: la semilla no influye")
    steps.append("fixture determinista (misma semilla -> mismos bytes)")

    # Exportación + determinismo + contenido de los headers
    r1 = run_export(m1, c1, out_path=tmp / "a" / "w.h", tv_path=tmp / "a" / "tv.h")
    r2 = run_export(m1, c1, out_path=tmp / "b" / "w.h", tv_path=tmp / "b" / "tv.h")
    for name in ("w.h", "tv.h"):
        _check((tmp / "a" / name).read_bytes() == (tmp / "b" / name).read_bytes(),
               f"exportación no determinista ({name})")
    _check((tmp / "a" / "w.h").read_text(encoding="ascii") == r1.weights_text, "el archivo escrito difiere")
    _check(r1.max_error < 1e-9, f"error de verificación inesperado: {r1.max_error}")
    _selftest_weights_text(r1)
    _selftest_vectors_text(r1)
    steps.append(f"exportación del fixture (state_dict): headers deterministas, {len(r1.vectors)} vectores, "
                 f"AE_MODEL_ID {r1.model_id}, error máx {r1.max_error:.2g}")

    # Otros formatos del mismo modelo -> mismo AE_MODEL_ID
    module, cfg = build_fixture(0)
    cfg_noacts = dict(cfg)
    del cfg_noacts["activations"]
    c_noacts = tmp / "cfg_noacts.json"
    write_atomic(c_noacts, json.dumps(cfg_noacts, indent=2))
    full_pt, ts_pt, ckpt_pt = tmp / "full.pt", tmp / "ts.pt", tmp / "ckpt.pt"
    torch.save(module, str(full_pt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.jit.script(module).save(str(ts_pt))
    torch.save({"epoch": 7, "model_state_dict": module.state_dict(), "loss": 0.25}, str(ckpt_pt))
    del warns[:]
    r_full = run_export(full_pt, c_noacts, out_path=tmp / "full" / "w.h", tv_path=tmp / "full" / "tv.h")
    _check(any("pickle" in w for w in warns), "falta el aviso de pickle para el módulo completo")
    r_ts = run_export(ts_pt, c_noacts, out_path=tmp / "ts" / "w.h", tv_path=tmp / "ts" / "tv.h")
    r_ck = run_export(ckpt_pt, c1, out_path=tmp / "ck" / "w.h", tv_path=None)
    for name, r in (("módulo completo", r_full), ("TorchScript", r_ts), ("checkpoint", r_ck)):
        _check(r.model_id == r1.model_id, f"{name}: AE_MODEL_ID {r.model_id} != {r1.model_id}")
    _check(r_full.activation_source == r_ts.activation_source == "modulos del modelo",
           "las activaciones no se introspectaron")
    _check(r_ck.kind.startswith("checkpoint['model_state_dict']"), f"checkpoint: formato {r_ck.kind}")
    steps.append("módulo completo (pickle), TorchScript y checkpoint -> mismo AE_MODEL_ID")

    # La verificación numérica rechaza activaciones incorrectas y activaciones funcionales
    _expect_error(lambda: run_export(ts_pt, c_noacts, "relu,identity,relu,identity",
                                     out_path=tmp / "bad" / "w.h", tv_path=tmp / "bad" / "tv.h"),
                  "verificación numérica FALLIDA", "TorchScript con --activations incorrectas")
    _check(not (tmp / "bad").exists(), "una exportación fallida escribió archivos")
    functional = _SelftestFunctionalAE()
    with torch.no_grad():
        for lin, src in zip([functional.l1, functional.l2, functional.l3, functional.l4],
                            [mod for mod in module.modules() if isinstance(mod, nn.Linear)]):
            lin.weight.copy_(src.weight)
            lin.bias.copy_(src.bias)
    fn_pt = tmp / "functional_ts.pt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.jit.script(functional).save(str(fn_pt))
    _expect_error(lambda: run_export(fn_pt, c_noacts, out_path=tmp / "bad" / "w.h", tv_path=None),
                  "activaciones funcionales", "F.relu en forward() sin activaciones explícitas")
    r_fn = run_export(fn_pt, c_noacts, "relu,relu,relu,identity", out_path=tmp / "fn" / "w.h", tv_path=None)
    _check(r_fn.max_error < 1e-9, "F.relu con --activations correctas no verifica")
    _check(not (tmp / "bad").exists(), "una exportación fallida escribió archivos")
    steps.append("verificación: rechaza --activations incorrectas y F.relu no declarado")

    # Validadores
    base = json.loads(c1.read_text(encoding="ascii"))

    def cfg_with(**changes: object) -> Dict[str, object]:
        out = json.loads(json.dumps(base))
        out.update(changes)
        return out

    std_zero = list(base["std"])
    std_zero[5] = 0.0
    _expect_error(lambda: parse_config(cfg_with(std=std_zero), 40, "cfg"), "'std' debe ser > 0", "std con un 0")
    _expect_error(lambda: parse_config(cfg_with(thresholds={"p95": 2.0, "p99": 1.0, "p99.9": 3.0}), 40, "cfg"),
                  "no monótonos", "umbrales no monótonos")
    _expect_error(lambda: parse_config(cfg_with(thresholds={"p95": 1.0, "p99": 2.0}), 40, "cfg"),
                  "p99.9", "falta p99.9")
    _expect_error(lambda: parse_config(cfg_with(mean=base["mean"][:39]), 40, "cfg"), "39 valores", "mean de 39")
    _expect_error(lambda: parse_config(cfg_with(score="mae"), 40, "cfg"), "score", "score distinto de mse")
    top = cfg_with(p95=1.0, p99=2.0, p999=3.0)
    del top["thresholds"]
    _check(parse_config(top, 40, "cfg").thresholds == (1.0, 2.0, 3.0), "umbrales en el nivel superior")
    layers = r1.model.layers
    _expect_error(lambda: validate_layers(layers[:3]), "exactamente 4", "3 capas")
    short = Layer("x", layers[3].weight[:39], layers[3].bias[:39])
    _expect_error(lambda: validate_layers(layers[:3] + [short]), "OUTPUT_DIM", "OUTPUT_DIM != INPUT_DIM")
    _expect_error(lambda: validate_layers([layers[0], layers[2], layers[1], layers[3]]), "cadena",
                  "cadena de dimensiones")
    sd_nan = {k: v.clone() for k, v in module.state_dict().items()}
    sd_nan["encoder.2.weight"][1, 3] = float("nan")
    _expect_error(lambda: layers_from_state_dict(sd_nan), "NaN", "peso NaN")
    nan_pt = tmp / "nan.pt"
    torch.save(sd_nan, str(nan_pt))
    _expect_error(lambda: run_export(nan_pt, c1, out_path=tmp / "bad" / "w.h", tv_path=None), "NaN",
                  "exportar un .pt con NaN")
    _check(not (tmp / "bad").exists(), "una exportación fallida escribió archivos")
    _expect_error(lambda: parse_activation("relu6", "test"), "desconocida", "activación desconocida")
    _expect_error(lambda: parse_activation("tanh:0.5", "test"), "no admite", "parámetro en tanh")
    _expect_error(lambda: parse_activation_list(["relu", "tanh", "relu"], "test"), "4 activaciones", "3 activaciones")
    _expect_error(lambda: run_export(m1, c1, "relu,foo,relu,identity", out_path=tmp / "bad" / "w.h",
                                     tv_path=None), "desconocida", "--activations con un nombre desconocido")
    _expect_error(lambda: run_export(m1, c_noacts, out_path=tmp / "bad" / "w.h", tv_path=None),
                  "--activations", "state_dict sin activaciones")
    _check(not (tmp / "bad").exists(), "una exportación fallida escribió archivos")
    _check(parse_activation(" LEAKY_RELU:0.2 ", "t") == Activation(ACT_LEAKY_RELU, float(np.float32(0.2)))
           and parse_activation("swish", "t").code == ACT_SILU
           and parse_activation("elu", "t").param == 1.0, "alias/parámetros de activación")
    steps.append("validadores: std 0, umbrales no monótonos, dims, NaN, activación desconocida, "
                 "state_dict sin activaciones")

    # x = MEAN por encima de T1: el nivel 0 se cubre partiendo del punto de menor score
    s_origin = r1.vectors[0].score
    c_low = tmp / "cfg_low.json"
    write_atomic(c_low, json.dumps(cfg_with(thresholds={"p95": s_origin / 2, "p99": 1.0, "p99.9": 2.0})))
    r_low = run_export(m1, c_low, out_path=tmp / "low" / "w.h", tv_path=tmp / "low" / "tv.h")
    counts = [sum(1 for v in r_low.vectors if v.level == k) for k in range(4)]
    _check(r_low.vectors[0].level == 1 and min(counts) >= 2, f"T1 < score de x = MEAN: niveles {counts}")
    steps.append(f"T1 por debajo del score de x = MEAN: niveles 0/1/2/3 cubiertos ({counts})")

    # Regla de niveles (igual que ae_level_from_score en C)
    th = r1.model.thresholds
    for k, t in enumerate(th, 1):
        _check(level_of(t, th) == k, f"score == T{k} debe dar nivel {k}")
        below = float(np.nextafter(np.float32(t), np.float32(0.0)))
        _check(level_of(below, th) == k - 1, f"score justo por debajo de T{k} debe dar {k - 1}")
    _check(level_of(0.0, th) == 0 and all(level_of(v, th) == 3 for v in (math.nan, math.inf, -math.inf)),
           "niveles de 0, NaN e infinitos")
    steps.append("regla de niveles: umbral exacto escala, nextafter por debajo, NaN/Inf -> 3")
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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="modelo .pt: state_dict, checkpoint, nn.Module o TorchScript")
    ap.add_argument("--config", type=Path, help="config.json con mean/std/umbrales (formato arriba)")
    ap.add_argument("--activations", help="4 activaciones separadas por comas (prioridad máxima), "
                                          "p.ej. relu,identity,relu,identity o leaky_relu:0.2,tanh,elu:0.5,identity")
    ap.add_argument("--out", type=Path,
                    help="header de pesos (def. firmware/detector_s3/include/autoencoder_weights.h)")
    tv = ap.add_mutually_exclusive_group()
    tv.add_argument("--test-vectors", type=Path,
                    help="header de vectores de prueba (def. firmware/detector_s3/test/autoencoder_test_vectors.h); "
                         "--out y --test-vectors se cambian juntos")
    tv.add_argument("--no-test-vectors", action="store_true",
                    help="no generar los vectores de prueba (quedarán desfasados respecto a los pesos)")
    ap.add_argument("--n-vectors", type=_int_range(MIN_N_VECTORS, MAX_N_VECTORS), default=DEFAULT_N_VECTORS,
                    help=f"número de vectores de prueba (def. {DEFAULT_N_VECTORS}, mín. {MIN_N_VECTORS})")
    ap.add_argument("--seed", type=_int_range(0, 2 ** 63 - 1), default=0,
                    help="semilla de los vectores de prueba, la verificación y el fixture (def. 0)")
    ap.add_argument("--make-fixture", type=Path, metavar="DIR",
                    help="escribir en DIR un modelo + config ALEATORIOS de prueba")
    ap.add_argument("--selftest", action="store_true", help="autotest en un directorio temporal (no toca el repo)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # UTF-8 también al redirigir a un pipe en Windows
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    export_mode = args.model is not None or args.config is not None
    modes = int(args.selftest) + int(args.make_fixture is not None) + int(export_mode)
    if modes == 0:
        ap.error("indicar --model y --config (o --make-fixture DIR, o --selftest)")
    if modes > 1:
        ap.error("elegir un solo modo: --model/--config, --make-fixture DIR o --selftest")
    if export_mode and (args.model is None or args.config is None):
        ap.error("--model y --config van juntos")
    out_path = args.out if args.out is not None else DEFAULT_WEIGHTS
    tv_path = None if args.no_test_vectors else (
        args.test_vectors if args.test_vectors is not None else DEFAULT_TEST_VECTORS)
    if export_mode and tv_path is not None and (
            _same_file(out_path, DEFAULT_WEIGHTS) != _same_file(tv_path, DEFAULT_TEST_VECTORS)):
        # Uno en el repo y el otro fuera: los headers del repo quedarían desfasados entre sí
        ap.error("--out y --test-vectors se cambian juntos (o usar --no-test-vectors): si solo uno sale "
                 "de su ruta por defecto, los pesos y los vectores del repo quedan desfasados")
    torch.set_num_threads(1)  # resultados reproducibles, independientes del número de núcleos
    try:
        if args.selftest:
            return selftest()
        if args.make_fixture is not None:
            model_path, config_path = make_fixture(args.make_fixture, args.seed)
            print_fixture_summary(model_path, config_path, args.seed)
            return 0
        result = run_export(args.model, args.config, args.activations, out_path, tv_path,
                            args.n_vectors, args.seed)
        print_summary(result)
        return 0
    except ExportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
