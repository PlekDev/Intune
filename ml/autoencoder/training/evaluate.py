# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "numpy", "pandas"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Evalúa el autoencoder, fija los umbrales de alerta y exporta config.json.

    uv run ml/autoencoder/training/evaluate.py [--normal ...] [--anomalous ...]

Umbrales (MSE de reconstrucción por ventana):
    MSE >= p95   -> nivel 1 (leve: bajar velocidad)
    MSE >= p99   -> nivel 2 (moderado: pausa)
    MSE >= p99.9 -> nivel 3 (severo: paro seguro)

Por defecto los percentiles salen del split de validación de los datos
NORMALES (el que no vio el entrenamiento): así p95 significa "5 % de falsas
alarmas de nivel >= 1 con el operador normal". Con --threshold-source anomalous
se toman de la distribución del dataset anómalo; en ese caso la mayoría de las
anomalías quedan por debajo de p95 y no disparan alerta.

El dataset anómalo se pasa por el modelo y se reporta qué fracción cae en cada
nivel. Si trae columna `label` (o `anomaly`/`is_anomaly`/`event`), se desglosa.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DEFAULT_MODEL, DEFAULT_NORMAL, ML_DIR, chrono_split, load_features  # noqa: E402
from models.autoencoder import BANDS, CHANNELS, HIDDEN, LATENT, Autoencoder, reconstruction_mse  # noqa: E402

DEFAULT_ANOMALOUS = ML_DIR / "data" / "anomalous_mock.csv"
DEFAULT_CONFIG = ML_DIR / "autoencoder" / "config.json"
PERCENTILES = {"p95": 95.0, "p99": 99.0, "p99.9": 99.9}
LABEL_COLUMNS = ("label", "anomaly", "is_anomaly", "event")


def score(model, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(((x - mean) / std).astype(np.float32))
    return reconstruction_mse(model, t).numpy()


def to_levels(mse: np.ndarray, thr: dict) -> np.ndarray:
    return ((mse >= thr["p95"]).astype(int) + (mse >= thr["p99"]) + (mse >= thr["p99.9"]))


def level_fractions(levels: np.ndarray) -> dict:
    n = max(1, len(levels))
    return {
        "level_ge_1": float((levels >= 1).sum() / n),
        "level_ge_2": float((levels >= 2).sum() / n),
        "level_3": float((levels == 3).sum() / n),
        "counts": [int((levels == lv).sum()) for lv in range(4)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--normal", type=Path, default=None, help="default: el CSV con el que se entrenó")
    ap.add_argument("--anomalous", type=Path, default=DEFAULT_ANOMALOUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--threshold-source", choices=["normal", "anomalous"], default="normal")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = Autoencoder()
    model.load_state_dict(ckpt["state_dict"])
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    normal_path = args.normal or Path(ckpt.get("normal_csv", DEFAULT_NORMAL))
    x_norm, cols_norm, _ = load_features(normal_path)
    _, x_val = chrono_split(x_norm)
    x_anom, cols_anom, df_anom = load_features(args.anomalous)
    if cols_norm != ckpt["features"] or cols_anom != ckpt["features"]:
        raise SystemExit("el orden de features de los CSV no coincide con el del modelo")

    mse_val = score(model, x_val, mean, std)
    mse_anom = score(model, x_anom, mean, std)
    source = mse_val if args.threshold_source == "normal" else mse_anom
    thr = {k: float(np.percentile(source, q)) for k, q in PERCENTILES.items()}

    lv_val = to_levels(mse_val, thr)
    lv_anom = to_levels(mse_anom, thr)
    report = {
        "normal_val": {"n": len(mse_val), "mse_median": float(np.median(mse_val)), **level_fractions(lv_val)},
        "anomalous": {"n": len(mse_anom), "mse_median": float(np.median(mse_anom)), **level_fractions(lv_anom)},
    }
    label_col = next((c for c in LABEL_COLUMNS if c in df_anom.columns), None)
    if label_col:
        report["anomalous_by_label"] = {
            str(lab): {"n": int(m.sum()), **level_fractions(lv_anom[m])}
            for lab in sorted(df_anom[label_col].unique(), key=str)
            for m in [(df_anom[label_col] == lab).to_numpy()]
        }

    config = {
        "project": "INTUNE",
        "node": "C2 Detector (ESP32-S3)",
        "model": {"arch": [len(mean), HIDDEN, LATENT, HIDDEN, len(mean)], "activation": "relu",
                  "output": "linear", "checkpoint": args.model.name,
                  "best_epoch": ckpt.get("best_epoch"), "best_val_mse": ckpt.get("best_val_mse")},
        "features": ckpt["features"],
        "channels": CHANNELS,
        "bands": BANDS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "score": "mean((x_norm - recon)^2) sobre las 40 features, x_norm = (x - mean) / std",
        "thresholds": thr,
        "levels": {"0": "normal", "1": "mild: slow down (mse >= p95)",
                   "2": "moderate: pause (mse >= p99)", "3": "severe: safe stop (mse >= p99.9)"},
        "threshold_source": f"{args.threshold_source} ({normal_path.name if args.threshold_source == 'normal' else args.anomalous.name})",
        "data": {"normal": str(normal_path), "anomalous": str(args.anomalous)},
        "evaluation": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    print(f"umbrales ({args.threshold_source}): " + "  ".join(f"{k}={v:.5f}" for k, v in thr.items()))
    for name, r in report.items():
        if name == "anomalous_by_label":
            for lab, rl in r.items():
                print(f"  {label_col}={lab:<12} n={rl['n']:5d}  >=1 {rl['level_ge_1']:6.1%}  >=2 {rl['level_ge_2']:6.1%}  3 {rl['level_3']:6.1%}")
            continue
        print(f"{name:<12} n={r['n']:5d}  MSE med {r['mse_median']:.4f}  >=1 {r['level_ge_1']:6.1%}  "
              f">=2 {r['level_ge_2']:6.1%}  3 {r['level_3']:6.1%}  niveles {r['counts']}")
    print(f"guardado {args.out}")


if __name__ == "__main__":
    main()
