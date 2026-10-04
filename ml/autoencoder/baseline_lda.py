"""Baseline LDA del Detector (C2): número de comparación y scorer de respaldo.

    uv run --project ml/autoencoder ml/autoencoder/baseline_lda.py --data <epochs.npz>

Features: epoch a 250 Hz con baseline [-200, 0) ms restado, promedio de cada
canal en 8 bins entre 150 y 700 ms -> 64 valores (canal mayor, bin menor), en
µV (no depende de la normalización de calibración).
Modelo: LDA con shrinkage (Ledoit-Wolf), error vs correcto, entrenado con
train + val del split cronológico y evaluado en test.
Exporta models/lda.json con w[64] y b: score = w · f + b (> 0 => más parecido
a error). En el S3 corre detrás de la misma lógica de alertas, con umbrales por
percentiles de los correctos de calibración.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

AE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AE_DIR))
import errp_pipeline as ep  # noqa: E402

DEFAULT_DATA = AE_DIR.parent / "data" / "epochs.npz"
OUT_PATH = AE_DIR / "models" / "lda.json"
BIN_MS = (150, 700)
N_BINS = 8
# Bordes en muestras a 250 Hz desde t = 0 (el S3 usa exactamente estos)
BIN_EDGES = np.round(np.linspace(BIN_MS[0], BIN_MS[1], N_BINS + 1) * ep.FS / 1000).astype(int)


def features(x: np.ndarray) -> np.ndarray:
    """[n, 8, 260] µV -> [n, 64]."""
    bc = ep.baseline_corrected(x)
    f = np.stack([bc[:, :, a:b].mean(axis=-1) for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])], axis=-1)
    return f.reshape(len(x), -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    d = ep.load_epochs(args.data)
    ok = ep.gate(d)
    tr, va, te = ep.chrono_split(len(d["X"]))
    fit_idx = np.concatenate([tr, va])
    fit_idx, te = fit_idx[ok[fit_idx]], te[ok[te]]
    f = features(d["X"])
    y = d["y"]

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(f[fit_idx], y[fit_idx])
    w = lda.coef_[0].astype(np.float32)
    b = float(lda.intercept_[0])
    s_te = f[te] @ w + b
    auc = float(roc_auc_score(y[te], s_te))
    thr = {k: float(np.percentile(s_te[y[te] == 0], q)) for k, q in {"T1": 90, "T2": 97, "T3": 99}.items()}
    det = {k: float((s_te[y[te] == 1] > v).mean()) for k, v in thr.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": "shrinkage LDA (lsqr, Ledoit-Wolf)",
        "features": {"layout": "canal mayor, bin menor", "channels": ep.CHANNELS,
                     "bin_ms": list(BIN_MS), "n_bins": N_BINS,
                     "bin_edges_samples_250hz": BIN_EDGES.tolist(),
                     "units": "µV, baseline [-200, 0) ms restado"},
        "w": w.tolist(), "b": b, "score": "w · f + b",
        "thresholds_offline": thr,
        "evaluation": {"auc": auc, "n_fit": int(len(fit_idx)), "n_test": int(len(te)),
                       "detection_rate_at": det},
        "data": str(args.data),
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"LDA: fit {len(fit_idx)} epochs, test {len(te)} ({y[te].mean():.0%} error)  AUC {auc:.3f}")
    print("detección de errores en test con umbrales p90/p97/p99 de correctos: "
          + "  ".join(f"{k} {v:.1%}" for k, v in det.items()))
    print(f"guardado {args.out}")


if __name__ == "__main__":
    main()
