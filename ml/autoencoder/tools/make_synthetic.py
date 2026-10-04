"""Genera epochs ErrP sintéticos en el formato de dataset de errp_pipeline.py.

    uv run --project ml/autoencoder ml/autoencoder/tools/make_synthetic.py

Solo para probar el pipeline mientras llegan los datos reales de C4. Simula
señal continua por sesión y la procesa como lo haría C4:
  - fondo 1/f con correlación espacial, offset DC grande (como el Unicorn crudo)
    y bursts de alfa posterior;
  - respuesta visual pequeña en TODAS las acciones (el AE debe aprenderla);
  - en acciones con error (~22 %): negatividad ~250 ms (Ne) y positividad
    ~400 ms (Pe), máximas en Fz/Cz, con jitter de latencia y amplitud por ensayo;
  - parpadeos aleatorios (frontales, >100 µV) para que el gate rechace epochs;
  - IIR causal exacto de C1 (sosfilt con estado inicial estacionario), corte
    [-220, +820) ms en cada pulso.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from errp_pipeline import CHANNELS, FS, MARGIN, POST, PRE, STORED_LEN, T0_INDEX  # noqa: E402

AE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = AE_DIR / "data" / "synthetic_epochs.npz"  # data/ está en .gitignore

#                 Fz   C3   Cz   C4   Pz   PO7  Oz   PO8
ERRP_W = np.array([1.0, 0.6, 1.0, 0.6, 0.5, 0.15, 0.1, 0.15])
VEP_W = np.array([0.2, 0.3, 0.3, 0.3, 0.5, 1.0, 1.0, 1.0])
BLINK_W = np.array([1.0, 0.35, 0.55, 0.35, 0.2, 0.05, 0.03, 0.05])
ALPHA_W = np.array([0.1, 0.2, 0.2, 0.2, 0.6, 1.0, 1.0, 1.0])


def pink_noise(rng, n_ch, n):
    spec = rng.normal(size=(n_ch, n // 2 + 1)) + 1j * rng.normal(size=(n_ch, n // 2 + 1))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec /= np.sqrt(np.maximum(f, 0.5))
    x = np.fft.irfft(spec, n)
    return x / x.std(axis=1, keepdims=True)


def gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def make_session(rng, n_actions, err_rate, bg_uv, errp_uv):
    onsets_s = 3.0 + np.cumsum(rng.uniform(1.5, 2.5, n_actions))  # 3 s iniciales de SETTLING
    n = int((onsets_s[-1] + 2.0) * FS)
    t = np.arange(n) / FS
    mix = np.eye(8) + 0.4 * rng.normal(size=(8, 8)) / np.sqrt(8)  # correlación espacial
    raw = bg_uv * (mix @ pink_noise(rng, 8, n))

    alpha_env = np.clip(signal.sosfiltfilt(signal.butter(2, 0.3, fs=FS, output="sos"), rng.normal(size=n)) * 6, 0, None)
    raw += 8.0 * ALPHA_W[:, None] * alpha_env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))

    y = (rng.random(n_actions) < err_rate).astype(np.int8)
    for onset, is_err in zip(onsets_s, y):
        i = int(onset * FS)
        tt = np.arange(-PRE - MARGIN, POST + 2 * MARGIN + 100) / FS  # hasta ~1.2 s
        seg = slice(i - PRE - MARGIN, i - PRE - MARGIN + len(tt))
        vep = -1.5 * gauss(tt, 0.15, 0.025) + 2.0 * gauss(tt, 0.22, 0.04)
        raw[:, seg] += VEP_W[:, None] * vep
        if is_err:
            amp = errp_uv * rng.uniform(0.5, 1.5)
            lat = rng.normal(0.0, 0.025)
            errp = -1.0 * gauss(tt, 0.25 + lat, 0.035) + 0.8 * gauss(tt, 0.40 + lat, 0.07)
            raw[:, seg] += amp * ERRP_W[:, None] * errp

    for blink_t in np.cumsum(rng.exponential(8.0, size=int(t[-1] / 4))):
        if blink_t >= t[-1] - 1:
            break
        i = int(blink_t * FS)
        bt = np.arange(0, int(0.4 * FS)) / FS
        raw[:, i:i + len(bt)] += rng.uniform(120, 250) * BLINK_W[:, None] * gauss(bt, 0.15, 0.05)

    raw *= rng.uniform(0.8, 1.2, size=(8, 1))   # contacto de electrodos por sesión
    raw += rng.uniform(-20000, 20000, size=(8, 1))  # offset DC del Unicorn crudo

    sos = signal.butter(2, [1, 15], btype="band", fs=FS, output="sos")
    zi = np.stack([signal.sosfilt_zi(sos) * raw[c, 0] for c in range(8)], axis=1)
    filt, _ = signal.sosfilt(sos, raw, axis=1, zi=zi)

    idx = (onsets_s * FS).astype(int)
    X = np.stack([filt[:, i - T0_INDEX:i - T0_INDEX + STORED_LEN] for i in idx]).astype(np.float32)
    return X, y, onsets_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--actions", type=int, default=300, help="acciones por sesión")
    ap.add_argument("--err-rate", type=float, default=0.22)
    ap.add_argument("--bg-uv", type=float, default=8.0, help="RMS del fondo antes del filtro")
    ap.add_argument("--errp-uv", type=float, default=6.0, help="amplitud media de la Ne")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    parts = [make_session(rng, args.actions, args.err_rate, args.bg_uv, args.errp_uv) for _ in range(args.sessions)]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    session = np.concatenate([np.full(len(p[1]), s, np.int32) for s, p in enumerate(parts)])
    onset_s = np.concatenate([p[2] for p in parts])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, session=session, onset_s=onset_s,
                        channels=np.array(CHANNELS), synthetic=np.array(True))
    print(f"{args.out}: {len(X)} epochs, {y.mean():.1%} con error, X {X.shape}, "
          f"|X| p99 {np.percentile(np.abs(X), 99):.1f} µV")


if __name__ == "__main__":
    main()
