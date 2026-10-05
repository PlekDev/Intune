"""Entrena el ErrP-AE (Keras) con el dataset de C4 (ml/data/FORMAT.md).

    uv run --project ml/autoencoder ml/autoencoder/training/train.py [sesiones.npz | carpeta ...]
        [--test-session NOMBRE] [--val-session NOMBRE] [--arch auto|conv|dense]

Por defecto lee ml/data/processed/*.npz. Split por sesión (errp_pipeline.split_sessions):
  train: correctas limpias no-calibración de las sesiones de entrenamiento, con
         augmentation nueva cada época (jitter ±1 muestra, ganancia ±10 %, ruido);
  val:   correctas limpias de la sesión de validación, para early stopping y elegir arquitectura;
  test:  una sesión entera, reservada para evaluate.py.
Cada época se normaliza con norm_mean/norm_std de SU sesión.

--arch auto entrena conv y dense y se queda con el de menor MSE en val (si el conv
sobreajusta, gana el dense). Guarda models/errp_ae.keras y models/errp_ae_meta.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE_DIR))
import errp_pipeline as ep  # noqa: E402

import keras  # noqa: E402
from models.errp_ae import BUILDERS  # noqa: E402

MODEL_PATH = AE_DIR / "models" / "errp_ae.keras"
META_PATH = AE_DIR / "models" / "errp_ae_meta.json"


class AugmentedEpochs(keras.utils.PyDataset):
    """Augmentation nueva en cada época; entrada = objetivo (autoencoder)."""

    def __init__(self, split, batch, seed, noise_std):
        super().__init__()
        self.split, self.batch, self.noise_std = split, batch, noise_std
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.split))

    def __len__(self):
        return int(np.ceil(len(self.split) / self.batch))

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        z = ep.to_model(ep.augment(self.split, idx, self.rng, self.noise_std))
        return z, z


def load_splits(paths, test=None, val=None):
    sessions = ep.load_sessions(paths)
    train, val_sp, s_test, test_idx, note = ep.split_sessions(sessions, test, val)
    return sessions, train, val_sp, s_test, test_idx, note


def train_one(arch, train, val, args):
    keras.utils.set_random_seed(args.seed)
    model = BUILDERS[arch]()
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay), loss="mse")
    z_val, z_tr = ep.to_model(val.z), ep.to_model(train.z)
    gen = AugmentedEpochs(train, args.batch, args.seed, args.noise)
    hist = model.fit(gen, validation_data=(z_val, z_val), epochs=args.epochs, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)])
    v = float(model.evaluate(z_val, z_val, verbose=0))
    t = float(model.evaluate(z_tr, z_tr, verbose=0))
    best = int(np.argmin(hist.history["val_loss"]))
    print(f"[{arch:5s}] params {model.count_params():6d}  epochs {len(hist.history['loss']):3d} (best {best})  "
          f"train MSE {t:.4f}  val MSE {v:.4f}  val/train {v / t:.2f}")
    return model, {"arch": arch, "params": model.count_params(), "best_epoch": best,
                   "train_mse": t, "val_mse": v, "overfit_ratio": v / t}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="*", type=Path, help="sesiones .npz o carpetas (default ml/data/processed)")
    ap.add_argument("--test-session")
    ap.add_argument("--val-session")
    ap.add_argument("--arch", choices=["auto", "conv", "dense"], default="auto")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.1, help="σ del ruido gaussiano (unidades z)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sessions, train, val, s_test, test_idx, note = load_splits(args.data, args.test_session, args.val_session)
    print(f"{len(sessions)} sesiones; {note}")
    print(f"train {len(train)} / val {len(val)} correctas limpias; test {s_test.name}: {len(test_idx)} acciones")

    archs = ["conv", "dense"] if args.arch == "auto" else [args.arch]
    results = [train_one(a, train, val, args) for a in archs]
    model, info = min(results, key=lambda r: r[1]["val_mse"])
    print(f"elegido: {info['arch']}")

    train_names = sorted({s.name for s in sessions} - {s_test.name}) or [s_test.name]  # 1 sesión
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    META_PATH.write_text(json.dumps({
        **info,
        "candidates": [r[1] for r in results],
        "data": [str(s.path) for s in sessions],
        "split": {"note": note, "test_session": s_test.name, "train_val_sessions": train_names,
                  "n_train": len(train), "n_val": len(val), "n_test_actions": int(len(test_idx))},
        "synthetic": all("synth" in s.name for s in sessions),
        # Normalización por defecto del S3 antes de calibrar: promedio de las sesiones de entrenamiento
        "mean": np.mean([s.mean for s in sessions if s.name in train_names], axis=0).tolist(),
        "std": np.mean([s.std for s in sessions if s.name in train_names], axis=0).tolist(),
        "hyper": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "data"},
    }, indent=2) + "\n")
    print(f"guardado {MODEL_PATH} y {META_PATH.name}")


if __name__ == "__main__":
    main()
