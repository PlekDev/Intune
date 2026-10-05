"""Evalúa el ErrP-AE (float e int8) simulando el S3 y exporta config.json.

    uv run --project ml/autoencoder ml/autoencoder/training/evaluate.py

Usa la sesión de test que eligió train.py (models/errp_ae_meta.json):
  umbrales: T1 = p90, T2 = p97, T3 = p99 del score de SUS épocas de calibración
            (is_calibration, limpias), como hará el S3 en vivo (FORMAT.md §6 regla 4);
  métricas: sobre sus acciones sin calibración, en orden de grabación:
            AUC, detección con 1 % de falsa alarma, tasas por nivel con la máquina de
            estados de CLAUDE.md, paros falsos por hora y lo mismo para el LDA.
Scorer principal = int8 (lo que corre en el S3) si existe el .tflite.
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
from baseline_lda import BIN_EDGES  # noqa: E402
from baseline_lda import OUT_PATH as LDA_PATH  # noqa: E402
from score import Scorer  # noqa: E402
from train import META_PATH  # noqa: E402

CONFIG_PATH = AE_DIR / "config.json"
PERCENTILES = {"T1": 90.0, "T2": 97.0, "T3": 99.0}


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


def evaluate_scorer(s: ep.Session, scores: np.ndarray, idx: np.ndarray) -> dict:
    """Umbrales con la calibración de la sesión; métricas sobre idx (acciones sin calibración)."""
    thr = {k: float(np.percentile(scores[s.cal_mask], q)) for k, q in PERCENTILES.items()}
    rej = s.rejected[idx]
    y = s.y[idx]
    sc = scores[idx]
    clean = ~rej
    levels = alert_levels(sc, rej, thr)
    corr, err = clean & (y == 0), clean & (y == 1)
    span = s.evt_counter[idx][s.evt_counter[idx] >= 0]
    hours = (span.max() - span.min()) / ep.FS / 3600 if len(span) > 1 else float("nan")
    false_stops = int(((levels == 3) & corr).sum())
    return {
        "thresholds": thr,
        "auc": float(roc_auc_score(y[clean], sc[clean])),
        "detection_at_1pct_fa": float((sc[err] > np.percentile(sc[corr], 99)).mean()),
        "correct": rates(levels[corr]),
        "error": rates(levels[err]),
        "rejected": int(rej.sum()),
        "false_stops_per_hour": false_stops / hours if hours > 0 else None,
        "eval_minutes": hours * 60,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=CONFIG_PATH)
    ap.add_argument("--n-golden", type=int, default=4, help="épocas de referencia para la prueba 12 en el S3")
    args = ap.parse_args()

    meta = json.loads(META_PATH.read_text())
    sessions = ep.load_sessions(meta["data"])
    _, _, s, test_idx, note = ep.split_sessions(sessions, meta["split"]["test_session"])
    sc = Scorer()
    r = sc.session(s)
    have_int8 = sc.int8 is not None
    main_key = "ae_int8" if have_int8 else "ae_float"

    report = {"split": note, "test_session": s.name, "main_scorer": main_key}
    for k in ("ae_float", "ae_int8", "lda"):
        if np.isfinite(r[k][~s.rejected]).all() and (k != "ae_int8" or have_int8) and (k != "lda" or sc.lda):
            report[k] = evaluate_scorer(s, r[k], test_idx)
    ok = np.flatnonzero(~s.rejected)
    if have_int8:
        report["float_int8_corr_test"] = float(np.corrcoef(r["ae_float"][ok], r["ae_int8"][ok])[0, 1])

    gold_c = [i for i in test_idx if not s.rejected[i] and s.y[i] == 0][:args.n_golden // 2]
    gold_e = [i for i in test_idx if not s.rejected[i] and s.y[i] == 1][:args.n_golden - len(gold_c)]
    golden = [{"label": int(s.y[i]), "x_uv": np.round(s.X[i], 5).tolist(),
               "mean": s.mean.tolist(), "std": s.std.tolist(),
               "input_norm": np.round(s.z([i])[0], 6).tolist(),
               "score_float": float(r["ae_float"][i]), "lda": float(r["lda"][i]) if sc.lda else None,
               **({"score_int8": float(r["ae_int8"][i])} if have_int8 else {})} for i in gold_c + gold_e]

    main_thr = report[main_key]["thresholds"]
    config = {
        "project": "INTUNE",
        "node": "C2 Detector (ESP32-S3)",
        "synthetic_data": meta.get("synthetic", False),
        "dataset_format": "ml/data/FORMAT.md (intune-c4-dataset-1.0)",
        "preprocessing": {
            "fs_hz": ep.FS, "channels": ep.CHANNELS,
            "epoch_ms": [-200, 800], "baseline_ms": [-200, 0], "model_window_ms": [0, 800],
            "decimation": {"factor": ep.DECIM, "method": "boxcar mean"},
            "input_shape": [ep.N_CH, ep.N_T],
            "gate": "FORMAT.md §3 (amplitud sobre X diezmado, gyro pico a pico por eje, flat std < 0.05 µV)",
        },
        "normalization": {
            "type": "z-score por canal con la calibración de la sesión",
            "mean": meta["mean"], "std": meta["std"],
            "note": "promedio de sesiones de entrenamiento; solo hasta que el S3 calibra (NVS)",
        },
        "model": {"arch": meta["arch"], "params": meta["params"], "candidates": meta.get("candidates"),
                  "tflite": meta.get("tflite"),
                  "score": "MSE(x_norm, recon) en float tras decuantizar, promedio sobre 8 x 40"},
        "thresholds": {**main_thr, "percentiles": PERCENTILES, "scorer": main_key,
                       "source": f"épocas de calibración de la sesión de test {s.name}; el S3 recalibra por sesión"},
        "lda": {"file": LDA_PATH.name, "bin_edges_samples_50hz": BIN_EDGES.tolist()},
        "evaluation": report,
        "golden": golden,
    }
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    print(note)
    print(f"test {s.name}: calibración {int(s.cal_mask.sum())} épocas, evaluadas {len(test_idx)} acciones")
    for k in ("ae_float", "ae_int8", "lda"):
        if k not in report:
            continue
        e = report[k]
        fs = e["false_stops_per_hour"]
        print(f"{k:8s} AUC {e['auc']:.3f}  det@1%FA {e['detection_at_1pct_fa']:5.1%}  "
              f"correctas >=1 {e['correct']['ge1']:5.1%} =3 {e['correct']['eq3']:5.1%} | "
              f"errores >=1 {e['error']['ge1']:5.1%} =3 {e['error']['eq3']:5.1%} | "
              f"paros falsos/h {fs if fs is None else round(fs, 1)} ({e['eval_minutes']:.0f} min)")
    if have_int8:
        print(f"correlación float/int8 en test: {report['float_int8_corr_test']:.4f}")
    print(f"guardado {args.out}")


if __name__ == "__main__":
    main()
