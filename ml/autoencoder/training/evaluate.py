"""Evalúa el ErrP-AE (float e int8), fija umbrales y exporta config.json.

    uv run --project ml/autoencoder ml/autoencoder/training/evaluate.py

El split de test (último 15 %, nunca visto en entrenamiento) se parte en dos
mitades cronológicas, como en una sesión real:
  calib: solo sus epochs correctos -> T1 = p90, T2 = p97, T3 = p99 del score
         (simula el bloque de calibración; en vivo el S3 los recalcula por sesión);
  eval:  correctos + errores -> tasas por nivel y máquina de estados de alertas.
El AUC (error vs correcto) usa todo el test. Score = MSE en espacio normalizado.
Usa el int8 (lo que corre en el S3) para umbrales y tasas si existe el .tflite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

AE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE_DIR))
sys.path.insert(0, str(AE_DIR / "training"))
import errp_pipeline as ep  # noqa: E402
from export_tflite import TFLITE_PATH, Int8Scorer  # noqa: E402
from train import META_PATH, MODEL_PATH  # noqa: E402

import keras  # noqa: E402

CONFIG_PATH = AE_DIR / "config.json"
LDA_PATH = AE_DIR / "models" / "lda.json"
PERCENTILES = {"T1": 90.0, "T2": 97.0, "T3": 99.0}
ACTIONS_PER_MIN = 20.0  # si el dataset no trae onset_s


def alert_levels(scores: np.ndarray, rejected: np.ndarray, thr: dict) -> np.ndarray:
    """Máquina de estados por acción (CLAUDE.md, Alert levels), sin confirmación del operador."""
    levels = np.zeros(len(scores), int)
    level, gt2_streak, rej_streak = 0, 0, 0
    for i, (s, rej) in enumerate(zip(scores, rejected)):
        if rej:  # EPOCH_REJECTED: se mantiene el nivel; 3 seguidos -> 2
            rej_streak += 1
            if rej_streak >= 3:
                level = max(level, 2)
            levels[i] = level
            continue
        rej_streak = 0
        if s > thr["T3"]:
            level = 3
        elif s > thr["T2"]:
            level = 3 if gt2_streak >= 1 else 2
        elif s > thr["T1"]:
            level = 1
        else:
            level = 0
        gt2_streak = gt2_streak + 1 if s > thr["T2"] else 0
        levels[i] = level
    return levels


def rates(levels: np.ndarray) -> dict:
    n = max(1, len(levels))
    return {"n": int(len(levels)), "ge1": float((levels >= 1).sum() / n),
            "ge2": float((levels >= 2).sum() / n), "eq3": float((levels == 3).sum() / n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=CONFIG_PATH)
    ap.add_argument("--n-golden", type=int, default=4, help="epochs de referencia para la prueba 12 en el S3")
    args = ap.parse_args()

    meta = json.loads(META_PATH.read_text())
    d = ep.load_epochs(Path(meta["data"]))
    mean, std = np.float32(meta["mean"]), np.float32(meta["std"])
    _, _, te = ep.chrono_split(len(d["X"]))
    ok = ep.gate(d)[te]
    y = d["y"][te]
    e = ep.normalize(ep.preprocess(d["X"][te]), mean, std)
    em = ep.to_model(e)

    model = keras.models.load_model(MODEL_PATH)
    s_float = ep.mse_score(em, model.predict(em, verbose=0))
    have_int8 = TFLITE_PATH.exists()
    s_int8 = Int8Scorer(TFLITE_PATH.read_bytes()).score(em) if have_int8 else None
    s_main = s_int8 if have_int8 else s_float

    half = len(te) // 2
    calib = np.arange(len(te)) < half
    calib_correct = calib & ok & (y == 0)
    thr = {k: float(np.percentile(s_main[calib_correct], q)) for k, q in PERCENTILES.items()}

    ev = ~calib
    levels = alert_levels(s_main[ev], ~ok[ev], thr)
    y_ev, ok_ev = y[ev], ok[ev]
    hours = (np.ptp(d["onset_s"][te][ev]) / 3600) if "onset_s" in d else ev.sum() / ACTIONS_PER_MIN / 60
    false_stops = int(((levels == 3) & (y_ev == 0) & ok_ev).sum())

    auc = {"float": float(roc_auc_score(y[ok], s_float[ok]))}
    if have_int8:
        auc["int8"] = float(roc_auc_score(y[ok], s_int8[ok]))
    report = {
        "auc": auc,
        "correct": rates(levels[(y_ev == 0) & ok_ev]),
        "error": rates(levels[(y_ev == 1) & ok_ev]),
        "rejected_epochs": int((~ok_ev).sum()),
        "false_stops_per_hour": false_stops / hours if hours > 0 else None,
        "eval_hours": float(hours),
        "n_calib_correct": int(calib_correct.sum()),
    }
    lda = json.loads(LDA_PATH.read_text()) if LDA_PATH.exists() else None
    if lda:
        report["lda_auc"] = lda["evaluation"]["auc"]

    golden_idx = [i for i in np.where(ok)[0] if y[i] == 0][:args.n_golden // 2] + \
                 [i for i in np.where(ok)[0] if y[i] == 1][:args.n_golden - args.n_golden // 2]
    golden = [{"label": int(y[i]), "input_norm": np.round(e[i], 6).tolist(),
               "score_float": float(s_float[i]),
               **({"score_int8": float(s_int8[i])} if have_int8 else {})} for i in golden_idx]

    config = {
        "project": "INTUNE",
        "node": "C2 Detector (ESP32-S3)",
        "synthetic_data": meta.get("synthetic", False),
        "preprocessing": {
            "fs_hz": ep.FS, "channels": ep.CHANNELS, "filter": ep.FILTER_SOS_DESIGN,
            "epoch_ms": [-200, 800], "baseline_ms": [-200, 0], "model_window_ms": [0, 800],
            "decimation": {"factor": ep.DECIM, "method": "boxcar mean"},
            "input_shape": [len(ep.CHANNELS), ep.N_T],
            "gate": {"peak_uv": ep.GATE_UV, "flat_std_uv": ep.FLAT_STD_UV,
                     "reject_flags": ["HELD", "GAP", "SETTLING", "OVERLAP"]},
        },
        "normalization": {
            "type": "z-score por canal", "mean": meta["mean"], "std": meta["std"],
            "note": "valores offline por defecto; el S3 los reemplaza con su bloque de calibración (NVS)",
        },
        "model": {
            "arch": meta["arch"], "params": meta["params"],
            "candidates": meta.get("candidates"),
            "tflite": meta.get("tflite"),
            "score": "MSE(x_norm, recon) en float tras decuantizar, promedio sobre 8 x 40",
        },
        "thresholds": {**thr, "percentiles": PERCENTILES,
                       "source": "correctos del bloque calib (test, no vistos en train); el S3 recalibra por sesión"},
        "alert_logic": {
            "0": "score <= T1", "1": "T1 < score <= T2",
            "2": "T2 < score <= T3, o 3 epochs rechazados seguidos",
            "3": "score > T3, o 2 epochs seguidos > T2, o fail-safe",
            "rejected": "EPOCH_REJECTED, mantiene el nivel previo",
        },
        "evaluation": report,
        "golden": golden,
    }
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    src = "int8" if have_int8 else "float"
    print(f"umbrales ({src}, {calib_correct.sum()} correctos de calib): "
          + "  ".join(f"{k}={v:.4f}" for k, v in thr.items()))
    print("AUC " + "  ".join(f"{k} {v:.3f}" for k, v in auc.items())
          + (f"  | LDA {report['lda_auc']:.3f}" if lda else "  | LDA (corre baseline_lda.py)"))
    for name in ("correct", "error"):
        r = report[name]
        print(f"{name:8s} n={r['n']:4d}  nivel>=1 {r['ge1']:6.1%}  >=2 {r['ge2']:6.1%}  =3 {r['eq3']:6.1%}")
    print(f"rechazados {report['rejected_epochs']}, paros falsos/h {report['false_stops_per_hour']:.1f} "
          f"({report['eval_hours'] * 60:.1f} min evaluados)")
    print(f"guardado {args.out}")


if __name__ == "__main__":
    main()
