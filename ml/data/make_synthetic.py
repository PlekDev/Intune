#!/usr/bin/env python3
"""
C4: genera una sesion FALSA con el mismo formato crudo que tools/recording/record_errp.py
(eeg.csv, events.csv, blocks.csv, status.csv, meta.json) para probar build_dataset.py y
ml/eval/ antes de tener grabaciones reales.

  python ml/data/make_synthetic.py [--seed 1] [--actions 300] [--latency-ms 0] [--name NOMBRE]

Simula lo que hace el puente (C1): el crudo r_* (con offset DC grande) pasa por el IIR causal
1-15 Hz (sosfilt, estado inicial sosfilt_zi*x0), huecos de 1 muestra = HELD, 2-25 = HELD|GAP
(relleno ZOH), > 25 = filas ausentes (salto de contador) + reinicio del filtro + SETTLING 2 s.
El ErrP sintetico (Fz/Cz, negativo ~250 ms y positivo ~350 ms) solo esta en las acciones
con label == error. Es una prueba de plomeria: NO es evidencia de que el ErrP real sea detectable.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "linux_probe"))
from design_iir import design_sos  # noqa: E402  (mismos coeficientes que el firmware)

FS = 250
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
F_HELD, F_GAP, F_SETTLING, F_FILTER_RESET, F_SESSION_START = 0x01, 0x02, 0x04, 0x08, 0x10
LINK_MAX_HOLD = 25
SETTLE = 500
ERRP_W = np.array([1.0, 0.4, 0.9, 0.4, 0.3, 0.0, 0.0, 0.0])  # peso espacial del ErrP
FILTER_TXT = "scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos'); sosfilt causal, zi = sosfilt_zi * x0"


def pink(rng, n, std, lo=0.5, hi=40.0):
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.zeros_like(f)
    m = (f >= lo) & (f <= hi)
    amp[m] = 1 / np.sqrt(f[m])
    x = np.fft.irfft(spec * amp, n)
    return x / x.std() * std


def bump(n, center, sigma, amp):
    t = np.arange(n)
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--actions", type=int, default=300)
    ap.add_argument("--error-rate", type=float, default=0.22)
    ap.add_argument("--latency-ms", type=float, default=0.0,
                    help="retardo del flanco de sincronia respecto al inicio real (BT + IIR); "
                         "build_dataset lo compensa con --latency-offset-ms")
    ap.add_argument("--errp-uv", type=float, default=10.0, help="amplitud del pico negativo")
    ap.add_argument("--subject", default="synthetic")
    ap.add_argument("--name", default=None, help="nombre de carpeta (por defecto <fecha>_<sujeto>)")
    ap.add_argument("--out", type=Path, default=HERE / "raw")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # ---- guion: bloques y luego acciones ----
    t = 6 * FS
    blocks, block_len = [], 30 * FS
    for name in ["eyes_open", "eyes_closed", "blinks", "jaw", "head"]:
        blocks.append((name, t, t + block_len))
        t += block_len + 5 * FS
    t += 5 * FS
    onsets, labels = [], []
    for _ in range(a.actions):
        onsets.append(t)
        labels.append("error" if rng.random() < a.error_rate else "correct")
        t += int(rng.uniform(2.5, 4.5) * FS)
    n = t + 5 * FS
    onsets = np.array(onsets)

    # ---- EEG crudo verdadero [n, 8] ----
    sd = rng.uniform(10, 20, 8)
    eeg = np.stack([pink(rng, n, sd[c]) for c in range(8)], axis=1)
    tt = np.arange(n) / FS
    alpha = np.sin(2 * np.pi * 10.2 * tt + rng.uniform(0, 6.28, 8)[:, None]).T * np.array([0, 0, 0, 0, 3, 8, 10, 8])
    amp_env = np.ones(n) * 0.5
    ec = [b for b in blocks if b[0] == "eyes_closed"][0]
    amp_env[ec[1]:ec[2]] = 3.0
    eeg += alpha * amp_env[:, None]

    gyr = rng.normal(0, 0.4, (n, 3)) + rng.normal(0, 1.0, 3)
    acc = rng.normal(0, 0.004, (n, 3)) + np.array([0.0, 0.0, 1.0])

    def blink(at, scale=1.0):
        w = int(0.3 * FS)
        if at < 0 or at + w >= n:
            return
        shape = np.sin(np.linspace(0, np.pi, w)) * rng.uniform(120, 200) * scale
        eeg[at:at + w] += shape[:, None] * np.array([1.0, 0.4, 0.5, 0.4, 0.2, 0.1, 0.05, 0.1])

    def jaw(at, dur):
        w = int(dur * FS)
        if at + w >= n:
            return
        eeg[at:at + w] += rng.normal(0, 70, (w, 8)) * np.hanning(w)[:, None]

    def head(at, dur):
        w = int(dur * FS)
        if at + w >= n:
            return
        ph = np.sin(np.linspace(0, 2 * np.pi, w)) * rng.uniform(60, 120)
        gyr[at:at + w] += ph[:, None] * rng.uniform(0.5, 1, 3)
        eeg[at:at + w] += (np.sin(np.linspace(0, np.pi, w)) * 150)[:, None] * rng.uniform(0.3, 1, 8)

    _, b0, b1 = [b for b in blocks if b[0] == "blinks"][0]
    for at in range(b0 + FS, b1 - FS, int(2.5 * FS)):
        blink(at)
    _, j0, j1 = [b for b in blocks if b[0] == "jaw"][0]
    for at in range(j0 + FS, j1 - 2 * FS, 4 * FS):
        jaw(at, 1.5)
    _, h0, h1 = [b for b in blocks if b[0] == "head"][0]
    for at in range(h0 + FS, h1 - 3 * FS, 4 * FS):
        head(at, 2.0)

    # ErrP (solo error) y artefactos ocasionales en acciones
    for on, lab in zip(onsets, labels):
        if lab == "error":
            g = rng.lognormal(0, 0.3)
            j = int(rng.uniform(-0.02, 0.02) * FS)
            wave = bump(n, on + int(0.25 * FS) + j, 0.04 * FS, -a.errp_uv * g) + \
                bump(n, on + int(0.35 * FS) + j, 0.06 * FS, 0.8 * a.errp_uv * g)
            eeg += wave[:, None] * ERRP_W
        r = rng.random()
        if r < 0.07:
            blink(on + int(rng.uniform(-0.3, 0.5) * FS))
        elif r < 0.10:
            jaw(on + int(rng.uniform(-0.2, 0.3) * FS), 1.0)
        elif r < 0.13:
            head(on + int(rng.uniform(-0.5, 0.0) * FS), 1.5)

    raw = eeg + rng.uniform(-1e5, 1e5, 8) + np.cumsum(rng.normal(0, 0.5, (n, 8)), axis=0) * 0.05
    # saturacion de un canal en una accion y canal plano 3 s en otra
    sat_on = onsets[a.actions // 7]
    raw[sat_on + 10:sat_on + 40, 3] = 750_000.0
    flat_on = onsets[a.actions * 3 // 10]
    raw[flat_on - 5 * FS:flat_on + 2 * FS, 6] = raw[flat_on - 5 * FS, 6]

    # ---- huecos del puente ----
    flags = np.zeros(n, np.uint8)
    keep = np.ones(n, bool)
    reset_at = [0]
    first_a = onsets[0] - 3 * FS
    pos = np.sort(rng.choice(np.arange(first_a, n - 3 * FS, FS), 26, replace=False))
    lens = [1] * 15 + list(rng.integers(2, LINK_MAX_HOLD + 1, 8)) + list(rng.integers(30, 200, 3))
    rng.shuffle(lens)
    for p, ln in zip(pos, lens):
        ln = int(ln)
        if ln <= LINK_MAX_HOLD:
            raw[p:p + ln] = raw[p - 1]
            flags[p:p + ln] |= F_HELD | (F_GAP if ln >= 2 else 0)
        else:
            keep[p:p + ln] = False
            reset_at.append(p + ln)

    # ---- IIR causal por segmento ----
    sos = design_sos()
    zi_unit = signal.sosfilt_zi(sos)
    filt = np.zeros_like(raw)
    bounds = sorted(set(reset_at)) + [n]
    for s, e in zip(bounds[:-1], bounds[1:]):
        e_eff = e
        idx = np.arange(s, e_eff)
        idx = idx[keep[idx]]
        seg = raw[idx]
        filt[idx], _ = signal.sosfilt(sos, seg, axis=0, zi=zi_unit[:, :, None] * seg[0])
        flags[s] |= F_FILTER_RESET
        flags[s:s + SETTLE] |= F_SETTLING
    flags[0] |= F_SESSION_START

    # ---- filas: solo las presentes ----
    rows = np.flatnonzero(keep)
    counter = rows + 1
    jitter = np.abs(rng.normal(0, 0.012, len(rows)))
    t0 = 1_700_000_000.0
    t_arr = t0 + counter * 0.004 + jitter
    df = pd.DataFrame({"t": t_arr, "counter": counter, "flags": flags[rows]})
    for c, name in enumerate(CH):
        df[f"f_{name}"] = filt[rows, c]
    for k, nm in enumerate("xyz"):
        df[f"acc_{nm}"] = acc[rows, k]
    for k, nm in enumerate("xyz"):
        df[f"gyr_{nm}"] = gyr[rows, k]
    for c, name in enumerate(CH):
        df[f"r_{name}"] = raw[rows, c]

    # ---- eventos ----
    lat = int(round(a.latency_ms / 4.0))
    ev = []
    for i, (on, lab) in enumerate(zip(onsets, labels)):
        ev.append(dict(action_id=i, label=lab, onset=on, status="ok", evt_flags=0))
    for i in rng.choice(np.arange(5, a.actions), 3, replace=False):
        ev[i]["status"] = "no_pulse"
    ev[int(rng.integers(5, a.actions))]["status"] = "arm_err"
    j = int(rng.integers(5, a.actions))
    if ev[j]["status"] == "ok":
        ev[j].update(status="no_t0", evt_flags=1)
    for i in (a.actions // 5, a.actions // 2):  # par solapado: accion manual 0.5 s despues
        ev.append(dict(action_id=a.actions + len(ev) - a.actions, label="manual",
                       onset=onsets[i] + FS // 2, status="overlap", evt_flags=2))
    ev.sort(key=lambda e: e["onset"])
    rec = []
    for seq, e in enumerate(ev):
        ok_pulse = e["status"] in ("ok", "overlap")
        on_t = t0 + (e["onset"] + 1) * 0.004
        rec.append(dict(
            action_id=e["action_id"], label=e["label"], t_sent=on_t - 0.05, t_onset_arm=on_t,
            t_done_arm=on_t + 2.0, evt_seq=seq,
            evt_counter=(e["onset"] + 1 + lat) if ok_pulse else "",
            evt_offset_us=int(rng.integers(50, 200)) if ok_pulse else "",
            evt_flags=e["evt_flags"], status=e["status"], arm_msg=""))
    events = pd.DataFrame(rec)

    # ---- escribir ----
    name = a.name or f"{dt.datetime.now():%Y%m%d_%H%M%S}_{a.subject}"
    d = a.out / name
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "eeg.csv", index=False, float_format="%.4f")
    events.to_csv(d / "events.csv", index=False, float_format="%.6f")
    pd.DataFrame([dict(block=b, counter_start=s + 1, counter_end=e) for b, s, e in blocks]) \
        .to_csv(d / "blocks.csv", index=False)
    secs = np.arange(0, int(n / FS), 1)
    pd.DataFrame(dict(t=t0 + secs, state=2, lost=0, reconnects=0, battery=80)).to_csv(d / "status.csv", index=False)
    (d / "meta.json").write_text(json.dumps(dict(
        sujeto=a.subject, semilla=a.seed, sintetico=True, filtro=FILTER_TXT,
        parametros=dict(actions=a.actions, error_rate=a.error_rate, latency_ms=a.latency_ms, errp_uv=a.errp_uv),
        resumen=dict(filas=int(len(df)), acciones=len(events))), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sesion sintetica en {d}  ({len(df)} muestras, {len(events)} eventos, "
          f"{labels.count('error')} errores de {a.actions})")


if __name__ == "__main__":
    main()
