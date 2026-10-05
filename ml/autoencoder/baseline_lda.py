"""Baseline LDA del Detector (C2): número de comparación y scorer de respaldo.

    uv run --project ml/autoencoder ml/autoencoder/baseline_lda.py [sesiones.npz | carpeta ...]
        [--test-session NOMBRE]

Features (CLAUDE.md): media de cada canal en 8 bins entre 150 y 700 ms -> 64 valores
(canal mayor, bin menor), sobre X [8, 40] de C4 en µV (baseline restado, 50 Hz; la
muestra j cubre [20 j, 20 j + 20) ms). Los bordes redondean 150-700 ms a la rejilla
de 20 ms: j = 8, 11, 14, 18, 21, 25, 28, 32, 35 (160-700 ms). No depende de la
normalización de calibración.
Modelo: LDA con shrinkage (Ledoit-Wolf), error vs correcto, entrenado con las
épocas limpias (y = 0/1) de las sesiones que no son de test; evaluado en la sesión
de test (sin calibración). Umbrales T1-T3 = p90/p97/p99 de los scores de las épocas
de calibración de la sesión de test, como en el S3.
Exporta models/lda.json con w[64] y b: score = w · f + b.
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

OUT_PATH = AE_DIR / "models" / "lda.json"
BIN_MS = (150, 700)
N_BINS = 8
BIN_EDGES = np.round(np.linspace(BIN_MS[0], BIN_MS[1], N_BINS + 1) / (1000 / 50)).astype(int)  # muestras a 50 Hz
PERCENTILES = {"T1": 90.0, "T2": 97.0, "T3": 99.0}


def features(x: np.ndarray) -> np.ndarray:
    """X [n, 8, 40] µV -> [n, 64]."""
    f = np.stack([x[:, :, a:b].mean(axis=-1) for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])], axis=-1)
    return f.reshape(len(x), -1).astype(np.float32)


def labeled_clean(s: ep.Session, idx=None) -> np.ndarray:
    m = (s.y >= 0) & ~s.rejected
    return np.flatnonzero(m) if idx is None else idx[m[idx]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="*", type=Path)
    ap.add_argument("--test-session")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    sessions = ep.load_sessions(args.data)
    _, _, s_test, test_idx, note = ep.split_sessions(sessions, args.test_session)
    if len(sessions) >= 3:
        fit = [(s, labeled_clean(s)) for s in sessions if s is not s_test]
    else:  # sin split por sesión: entrenar con lo que no es test de cada sesión
        fit = [(s, labeled_clean(s, np.setdiff1d(np.arange(len(s.y)), test_idx))) for s in sessions]
    f_fit = np.concatenate([features(s.X[i]) for s, i in fit])
    y_fit = np.concatenate([s.y[i] for s, i in fit])

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(f_fit, y_fit)
    w = lda.coef_[0].astype(np.float32)
    b = float(lda.intercept_[0])

    te = labeled_clean(s_test, test_idx)
    s_te = features(s_test.X[te]) @ w + b
    y_te = s_test.y[te]
    s_cal = features(s_test.X[s_test.cal_mask]) @ w + b
    thr = {k: float(np.percentile(s_cal, q)) for k, q in PERCENTILES.items()}
    auc = float(roc_auc_score(y_te, s_te))
    det = {k: float((s_te[y_te == 1] > v).mean()) for k, v in thr.items()}
    fa = {k: float((s_te[y_te == 0] > v).mean()) for k, v in thr.items()}
    det_at_1pct = float((s_te[y_te == 1] > np.percentile(s_te[y_te == 0], 99)).mean())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": "shrinkage LDA (lsqr, Ledoit-Wolf)",
        "features": {"layout": "canal mayor, bin menor", "channels": ep.CHANNELS,
                     "bin_ms": list(BIN_MS), "n_bins": N_BINS,
                     "bin_edges_samples_50hz": BIN_EDGES.tolist(),
                     "input": "X [8, 40] µV de C4 (baseline restado, 50 Hz)"},
        "w": w.tolist(), "b": b, "score": "w · f + b",
        "thresholds_test_calibration": thr,
        "evaluation": {"auc": auc, "detection_at_1pct_fa": det_at_1pct, "detection_at": det,
                       "false_alarm_at": fa, "n_fit": int(len(y_fit)), "n_test": int(len(te)),
                       "test_session": s_test.name, "split": note},
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"LDA: fit {len(y_fit)} épocas ({y_fit.mean():.0%} error), test {s_test.name} {len(te)} épocas  "
          f"AUC {auc:.3f}  detección @1% FA {det_at_1pct:.1%}")
    print("con umbrales de calibración: " + "  ".join(
        f"{k} det {det[k]:.1%} / FA {fa[k]:.1%}" for k in thr))
    print(f"guardado {args.out}")


if __name__ == "__main__":
    main()
