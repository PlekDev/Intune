"""Scores por acción del Detector (C2) para cualquier sesión de C4: lo que usa ml/eval/.

    uv run --project ml/autoencoder ml/autoencoder/score.py ml/data/processed/<sesion>.npz [...] \
        --model runs/r1/model.keras --tflite runs/r1/model.tflite [--lda ml/autoencoder/models/lda.json] \
        [--out scores.csv]

model.keras sale de train_errp_ae.py; model.tflite de ml/c_exporter/export_to_c.py --tflite-out
(los mismos bytes que AE_MODEL_TFLITE del S3). Por acción escribe: session, action_id, y,
rejected, is_calibration, evt_counter, ae_float, ae_int8, lda. Las rechazadas llevan NaN (el S3
no las puntúa).
  ae_float: MSE entre la época normalizada (norm_mean/norm_std de su sesión) y la reconstrucción
            del modelo Keras;
  ae_int8:  la misma cadena que ae_infer en el S3 (cuantizar -> modelo int8 -> decuantizar -> MSE)
            con la emulación entera de TFLite Micro de export_to_c.py (TflmReference): bit a bit
            con el S3;
  lda:      w · f + b con models/lda.json (features sobre X en µV).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

AE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AE_DIR))
sys.path.insert(0, str(AE_DIR.parent / "c_exporter"))
import errp_ae as ae  # noqa: E402
import errp_pipeline as ep  # noqa: E402
from baseline_lda import OUT_PATH as LDA_PATH  # noqa: E402
from baseline_lda import features as lda_features  # noqa: E402


class Scorer:
    def __init__(self, model_path: Path, tflite_path: Path | None, lda_path: Path | None):
        keras = ae.keras_module()
        self.model = keras.models.load_model(model_path, compile=False)
        self.emu = self.info = None
        if tflite_path:
            from export_to_c import TflmReference, inspect_tflite
            buf = Path(tflite_path).read_bytes()
            self.info = inspect_tflite(buf)
            self.emu = TflmReference(buf)
        self.lda = json.loads(Path(lda_path).read_text()) if lda_path and Path(lda_path).exists() else None

    def session(self, s: ep.Session) -> dict:
        """Scores de todas las acciones de la sesión; NaN en las rechazadas."""
        from export_to_c import int8_chain
        n = len(s.y)
        out = {k: np.full(n, np.nan, np.float32) for k in ("ae_float", "ae_int8", "lda")}
        ok = np.flatnonzero(~s.rejected)
        if len(ok):
            z = ae.normalize(s.X[ok], s.mean, s.std)
            out["ae_float"][ok] = ae.score(z, np.asarray(self.model.predict_on_batch(z[..., None])))
            if self.emu:
                out["ae_int8"][ok] = [int8_chain(zi, self.emu, self.info).score for zi in z]
            if self.lda:
                out["lda"][ok] = lda_features(s.X[ok]) @ np.float32(self.lda["w"]) + np.float32(self.lda["b"])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", type=Path, help="sesiones .npz de C4 o carpetas")
    ap.add_argument("--model", type=Path, required=True, help="model.keras de train_errp_ae.py")
    ap.add_argument("--tflite", type=Path, help="model.tflite de export_to_c.py --tflite-out (para ae_int8)")
    ap.add_argument("--lda", type=Path, default=LDA_PATH)
    ap.add_argument("--out", type=Path, default=Path("scores.csv"))
    args = ap.parse_args()

    sc = Scorer(args.model, args.tflite, args.lda)
    rows = 0
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "action_id", "y", "rejected", "is_calibration", "evt_counter",
                    "ae_float", "ae_int8", "lda"])
        for s in ep.load_sessions(args.data):
            with np.load(s.path, allow_pickle=False) as d:
                action_id = d["action_id"]
            r = sc.session(s)
            for k in range(len(s.y)):
                w.writerow([s.name, int(action_id[k]), int(s.y[k]), int(s.rejected[k]), int(s.is_cal[k]),
                            int(s.evt_counter[k]), *(f"{r[c][k]:.6g}" for c in ("ae_float", "ae_int8", "lda"))])
                rows += 1
    print(f"{rows} acciones -> {args.out}")


if __name__ == "__main__":
    main()
