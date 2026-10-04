"""Entrena el ErrP-AE (Keras) SOLO con epochs de acciones correctas.

    uv run --project ml/autoencoder ml/autoencoder/training/train.py --data <epochs.npz>

Split cronológico 70/15/15 sobre todos los epochs (errp_pipeline.chrono_split):
  train: correctos aceptados por el gate, con augmentation nueva cada época
         (jitter ±20 ms, ganancia por canal ±10 %, ruido gaussiano);
  val:   correctos sin augmentation, para early stopping;
  test:  correctos + errores, reservado para evaluate.py (umbrales y AUC).
Normalización z-score por canal con mean/std de los correctos de train.

--arch auto entrena conv y dense y se queda con el de menor MSE en val
(si el conv sobreajusta, gana el dense). Guarda models/errp_ae.keras y
models/errp_ae_meta.json.
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

DEFAULT_DATA = AE_DIR.parent / "data" / "epochs.npz"
MODEL_PATH = AE_DIR / "models" / "errp_ae.keras"
META_PATH = AE_DIR / "models" / "errp_ae_meta.json"


class AugmentedEpochs(keras.utils.PyDataset):
    """Augmentation nueva en cada época; entrada = objetivo (autoencoder)."""

    def __init__(self, x_raw, mean, std, batch, seed, noise_std):
        super().__init__()
        self.x_raw, self.mean, self.std = x_raw, mean, std
        self.batch, self.noise_std = batch, noise_std
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.x_raw))

    def __len__(self):
        return int(np.ceil(len(self.x_raw) / self.batch))

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        e = ep.to_model(ep.augment(self.x_raw[idx], self.rng, self.mean, self.std, self.noise_std))
        return e, e


def load_splits(path: Path) -> dict:
    d = ep.load_epochs(path)
    ok = ep.gate(d)
    tr, va, te = ep.chrono_split(len(d["X"]))
    correct = (d["y"] == 0) & ok
    s = {"d": d, "ok": ok, "train": tr[correct[tr]], "val": va[correct[va]], "test": te[ok[te]]}
    e_train = ep.preprocess(d["X"][s["train"]])
    s["mean"], s["std"] = ep.channel_stats(e_train)
    return s


def train_one(arch, s, args):
    keras.utils.set_random_seed(args.seed)
    model = BUILDERS[arch]()
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay), loss="mse")
    x = s["d"]["X"]
    e_val = ep.to_model(ep.normalize(ep.preprocess(x[s["val"]]), s["mean"], s["std"]))
    e_tr = ep.to_model(ep.normalize(ep.preprocess(x[s["train"]]), s["mean"], s["std"]))
    gen = AugmentedEpochs(x[s["train"]], s["mean"], s["std"], args.batch, args.seed, args.noise)
    hist = model.fit(gen, validation_data=(e_val, e_val), epochs=args.epochs, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)])
    val = float(model.evaluate(e_val, e_val, verbose=0))
    train_clean = float(model.evaluate(e_tr, e_tr, verbose=0))
    best_epoch = int(np.argmin(hist.history["val_loss"]))
    print(f"[{arch:5s}] params {model.count_params():6d}  epochs {len(hist.history['loss']):3d} "
          f"(best {best_epoch})  train MSE {train_clean:.4f}  val MSE {val:.4f}  val/train {val / train_clean:.2f}")
    return model, {"arch": arch, "params": model.count_params(), "best_epoch": best_epoch,
                   "train_mse": train_clean, "val_mse": val, "overfit_ratio": val / train_clean}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--arch", choices=["auto", "conv", "dense"], default="auto")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.1, help="σ del ruido gaussiano (unidades z)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s = load_splits(args.data)
    d = s["d"]
    print(f"{args.data}: {len(d['X'])} epochs, gate rechaza {(~s['ok']).sum()}, "
          f"train {len(s['train'])} / val {len(s['val'])} correctos, test {len(s['test'])} (con errores)")

    archs = ["conv", "dense"] if args.arch == "auto" else [args.arch]
    results = [train_one(a, s, args) for a in archs]
    model, info = min(results, key=lambda r: r[1]["val_mse"])
    print(f"elegido: {info['arch']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    META_PATH.write_text(json.dumps({
        **info,
        "candidates": [r[1] for r in results],
        "data": str(args.data),
        "synthetic": bool(d.get("synthetic", False)),
        "mean": s["mean"].tolist(),
        "std": s["std"].tolist(),
        "split": {"fractions": [0.70, 0.15, 0.15], "n_train": len(s["train"]),
                  "n_val": len(s["val"]), "n_test": len(s["test"])},
        "hyper": {k: v for k, v in vars(args).items() if k not in ("data",)},
    }, indent=2) + "\n")
    print(f"guardado {MODEL_PATH} y {META_PATH.name}")


if __name__ == "__main__":
    main()
