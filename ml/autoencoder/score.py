"""Scores por acción del Detector (C2) para cualquier sesión de C4: lo que usa ml/eval/.

    uv run --project ml/autoencoder ml/autoencoder/score.py ml/data/processed/<sesion>.npz [...] \
        [--out scores.csv]

Por acción escribe: session, action_id, y, rejected, is_calibration, evt_counter,
ae_float, ae_int8, lda. Las épocas rechazadas llevan NaN (el S3 no las puntúa).
  ae_*: MSE entre la época normalizada (norm_mean/norm_std de su sesión) y su
        reconstrucción; ae_int8 cuantiza la entrada y decuantiza la salida como el S3.
  lda:  w · f + b con models/lda.json (features sobre X en µV).
Necesita models/errp_ae.keras y, para ae_int8, models/errp_ae_int8.tflite.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

AE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AE_DIR))
sys.path.insert(0, str(AE_DIR / "training"))
import errp_pipeline as ep  # noqa: E402
from baseline_lda import OUT_PATH as LDA_PATH  # noqa: E402
from baseline_lda import features as lda_features  # noqa: E402

MODEL_PATH = AE_DIR / "models" / "errp_ae.keras"
TFLITE_PATH = AE_DIR / "models" / "errp_ae_int8.tflite"


class Scorer:
    def __init__(self, model_path=MODEL_PATH, tflite_path=TFLITE_PATH, lda_path=LDA_PATH):
        import keras  # import tardío: TensorFlow tarda en cargar
        self.model = keras.models.load_model(model_path)
        self.int8 = None
        if Path(tflite_path).exists():
            from export_tflite import Int8Scorer
            self.int8 = Int8Scorer(Path(tflite_path).read_bytes())
        self.lda = json.loads(Path(lda_path).read_text()) if Path(lda_path).exists() else None

    def session(self, s: ep.Session) -> dict:
        """Scores de todas las acciones de la sesión; NaN en las rechazadas."""
        n = len(s.y)
        out = {k: np.full(n, np.nan, np.float32) for k in ("ae_float", "ae_int8", "lda")}
        ok = np.flatnonzero(~s.rejected)
        if len(ok):
            z = ep.to_model(s.z(ok))
            out["ae_float"][ok] = ep.mse_score(z, self.model.predict(z, verbose=0))
            if self.int8:
                out["ae_int8"][ok] = self.int8.score(z)
            if self.lda:
                out["lda"][ok] = lda_features(s.X[ok]) @ np.float32(self.lda["w"]) + np.float32(self.lda["b"])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("scores.csv"))
    args = ap.parse_args()

    sc = Scorer()
    rows = 0
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "action_id", "y", "rejected", "is_calibration", "evt_counter",
                    "ae_float", "ae_int8", "lda"])
        for s in ep.load_sessions(args.data):
            d = np.load(s.path, allow_pickle=False)
            r = sc.session(s)
            for k in range(len(s.y)):
                w.writerow([s.name, int(d["action_id"][k]), int(s.y[k]), int(s.rejected[k]), int(s.is_cal[k]),
                            int(s.evt_counter[k]), *(f"{r[c][k]:.6g}" for c in ("ae_float", "ae_int8", "lda"))])
                rows += 1
    print(f"{rows} acciones -> {args.out}")


if __name__ == "__main__":
    main()
