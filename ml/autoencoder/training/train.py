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
"""Entrena el autoencoder del Detector (C2) SOLO con datos normales.

    uv run ml/autoencoder/training/train.py [--normal ml/data/normal_mock.csv]

CSV esperado (lo genera C4 en ml/data/): una fila por ventana, 40 columnas de
features. Si existen las columnas <canal>_<banda> (Fz_delta ... PO8_gamma) se
usan en ese orden; si no, se toman las columnas numéricas que no sean de
metadatos y deben ser exactamente 40.

Split train/val cronológico (último 20 % = val): las ventanas se traslapan y un
split aleatorio filtraría muestras casi idénticas a validación.

Guarda en el checkpoint: pesos del mejor epoch (val loss), mean/std de train y
el orden de features. evaluate.py lo usa para generar config.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ML_DIR = Path(__file__).resolve().parents[2]  # .../ml
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR / "autoencoder"))
from models.autoencoder import FEATURE_NAMES, N_FEATURES, Autoencoder  # noqa: E402

DEFAULT_NORMAL = ML_DIR / "data" / "normal_mock.csv"
DEFAULT_MODEL = ML_DIR / "autoencoder" / "models" / "autoencoder_best.pt"
META_COLUMNS = {"t", "time", "timestamp", "counter", "window", "label", "anomaly",
                "is_anomaly", "event", "session", "subject", "split"}
VAL_FRACTION = 0.2


def load_features(path: Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Devuelve (X float32 [n, 40], nombres de columnas, DataFrame original)."""
    df = pd.read_csv(path)
    if all(name in df.columns for name in FEATURE_NAMES):
        cols = FEATURE_NAMES
    else:
        cols = [c for c in df.select_dtypes("number").columns if c.lower() not in META_COLUMNS]
        if len(cols) != N_FEATURES:
            raise SystemExit(f"{path}: se esperaban {N_FEATURES} columnas de features, hay {len(cols)}: {cols}")
    x = df[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise SystemExit(f"{path}: hay NaN/inf en las features")
    return x, list(cols), df


def chrono_split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(len(x) * VAL_FRACTION))
    return x[:-n_val], x[-n_val:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25, help="epochs sin mejorar val antes de parar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, cols, _ = load_features(args.normal)
    x_train, x_val = chrono_split(x)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0  # feature constante: no escalar
    print(f"normal: {len(x)} ventanas ({len(x_train)} train / {len(x_val)} val), {len(cols)} features")

    t_train = torch.from_numpy((x_train - mean) / std)
    t_val = torch.from_numpy((x_val - mean) / std)

    model = Autoencoder()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(t_train))
        train_loss = 0.0
        for i in range(0, len(perm), args.batch):
            xb = t_train[perm[i:i + args.batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(t_train)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(t_val), t_val).item()
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  best {best_val:.5f}@{best_epoch}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop en epoch {epoch} (sin mejora desde {best_epoch})")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": cols,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "normal_csv": str(args.normal),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "seed": args.seed,
    }, args.out)
    print(f"guardado {args.out} (val MSE {best_val:.5f}, epoch {best_epoch})")


if __name__ == "__main__":
    main()
