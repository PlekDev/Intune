#!/usr/bin/env python3
"""
C4: sesion cruda (tools/recording/record_errp.py) -> un .npz por sesion en ml/data/processed/.

Implementa EXACTAMENTE la cadena "Signal chain" de CLAUDE.md, la misma del S3:
  t = 0 en evt_counter - EVENT_LATENCY_OFFSET  ->  ventana [-200, +800) ms sobre f_*
  (ya filtrado por el puente, NO se vuelve a filtrar)  ->  baseline = media por canal de
  [-200, 0) ms  ->  diezmado x5 por media de bloques sobre 0-800 ms  ->  X [8, 40]
  ->  puerta de artefactos  ->  mean/std por canal de las epocas de calibracion (aparte).

Uso:
  python ml/data/build_dataset.py                       # todas las sesiones de ml/data/raw/
  python ml/data/build_dataset.py ml/data/raw/<sesion>  # una sesion
  python ml/data/build_dataset.py <sesion> --suggest-gyro   # propone GATE_GYRO con los bloques
Contrato completo: ml/data/FORMAT.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_VERSION = "intune-c4-dataset-1.0"

# ---- constantes de la cadena (no son parametros: cambiarlas rompe el modelo en vivo) ----
FS_IN = 250
FS_OUT = 50
DECIMATION = 5
PRE_SAMPLES = 50        # -200 ms
POST_SAMPLES = 200      # +800 ms (muestras 0..199)
WINDOW_MS = (-200, 800)
BASELINE_MS = (-200, 0)
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
N_CH = 8
N_T = POST_SAMPLES // DECIMATION  # 40
SAMPLE_MS = 4.0

# flags de link_protocol.h
F_HELD, F_GAP, F_SETTLING, F_FILTER_RESET, F_SESSION_START, F_UNFILTERED = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
F_REJECT = F_HELD | F_GAP | F_SETTLING | F_FILTER_RESET

# ---- parametros por defecto (TUNE) ----
EVENT_LATENCY_OFFSET_MS = 0.0   # hasta la prueba 10
GATE_UV = 100.0                 # |amplitud| tras IIR, baseline y diezmado
GATE_GYRO = 30.0                # deg/s, rango pico a pico por eje en la ventana (ver --suggest-gyro)
FLAT_STD_UV = 0.05              # canal plano: std de la epoca < esto
ADC_SAT_UV = 740_000.0          # |r_*| >= esto = saturacion (tope del ADC ~750 000 uV)
OVERLAP_S = 1.0
N_CAL = 80                      # acciones correctas y limpias iniciales = calibracion
MIN_CAL = 20                    # menos que esto: norm_valid = False
STD_FLOOR = 1e-3

REASONS = ["status", "counter", "flags", "unfiltered", "saturated", "flat", "amplitude", "gyro", "overlap"]


def _cols(prefix):
    return [f"{prefix}_{c}" for c in CH]


def load_session(path: Path):
    eeg = pd.read_csv(path / "eeg.csv")
    ev = pd.read_csv(path / "events.csv")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8")) if (path / "meta.json").exists() else {}
    blocks = pd.read_csv(path / "blocks.csv") if (path / "blocks.csv").exists() else None
    return eeg, ev, meta, blocks


def gyro_metric(gyr_win: np.ndarray) -> float:
    """Rango pico a pico por eje, maximo de los 3. No depende del sesgo del giroscopio."""
    return float(np.max(gyr_win.max(axis=0) - gyr_win.min(axis=0)))


def suggest_gyro(eeg: pd.DataFrame, blocks: pd.DataFrame):
    """Propone GATE_GYRO: media geometrica entre el p99 en reposo (eyes_open) y el mediana del bloque head."""
    counter = eeg["counter"].to_numpy(np.int64)
    gyr = eeg[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(np.float64)
    win = PRE_SAMPLES + POST_SAMPLES

    def metrics(block):
        rows = blocks[blocks["block"] == block]
        out = []
        for _, b in rows.iterrows():
            idx = np.flatnonzero((counter >= b.counter_start) & (counter <= b.counter_end))
            for s in range(idx[0], idx[-1] - win, win // 2) if len(idx) > win else []:
                out.append(gyro_metric(gyr[s:s + win]))
        return np.array(out)

    rest, head = metrics("eyes_open"), metrics("head")
    if len(rest) == 0 or len(head) == 0:
        return None
    r99, h50 = np.percentile(rest, 99), np.percentile(head, 50)
    return dict(rest_p99=r99, head_p50=h50, suggested=float(np.sqrt(r99 * h50)), n_rest=len(rest), n_head=len(head))


def build_session(path: Path, latency_ms=EVENT_LATENCY_OFFSET_MS, gate_uv=GATE_UV, gate_gyro=GATE_GYRO,
                  n_cal=N_CAL):
    eeg, ev, meta, _ = load_session(path)
    counter = eeg["counter"].to_numpy(np.int64)
    flags = eeg["flags"].to_numpy(np.int64)
    f = eeg[_cols("f")].to_numpy(np.float64)
    r = eeg[_cols("r")].to_numpy(np.float64)
    gyr = eeg[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(np.float64)
    t_arr = eeg["t"].to_numpy(np.float64)
    n_rows = len(eeg)
    has_raw = bool(np.isfinite(r).any())   # r_* solo existe si el puente tiene EEG_RAW (Kconfig)

    lat_samples = int(round(latency_ms / SAMPLE_MS))
    n = len(ev)
    evc = pd.to_numeric(ev["evt_counter"], errors="coerce").to_numpy(np.float64)
    status = ev["status"].astype(str).to_numpy()
    evt_flags = pd.to_numeric(ev["evt_flags"], errors="coerce").fillna(0).astype(int).to_numpy()
    onset_arm = pd.to_numeric(ev["t_onset_arm"], errors="coerce").to_numpy(np.float64)
    t_sent = pd.to_numeric(ev["t_sent"], errors="coerce").to_numpy(np.float64)

    # Tramos de sesión Unicorn: tras una reconexión el contador vuelve a 1 y se REPITE dentro de
    # eeg.csv (también tras un salto > 10 s; ver respuesta de C1). Cada fila y cada evento se asignan
    # a su tramo por la hora de la PC; contador y overlap solo se comparan dentro del mismo tramo.
    ss_rows = np.flatnonzero(flags & F_SESSION_START)
    seg_row = np.cumsum((flags & F_SESSION_START) != 0)            # 1, 2, ... (0 = antes del primero)
    t_ev = np.where(np.isfinite(onset_arm), onset_arm, t_sent)
    seg_ev = np.searchsorted(t_arr[ss_rows], t_ev, side="right")   # tramo vigente al lanzar la acción

    X = np.zeros((n, N_CH, N_T), np.float32)
    W = np.zeros((n, N_CH, PRE_SAMPLES + POST_SAMPLES), np.float32)  # ventana f_* cruda [-200, +800)
    reasons = [[] for _ in range(n)]
    max_abs = np.full(n, np.nan, np.float32)
    gyro_m = np.full(n, np.nan, np.float32)

    for k in range(n):
        rs = reasons[k]
        if status[k] != "ok":
            rs.append("status")
        if not np.isfinite(evc[k]):
            if "status" not in rs:
                rs.append("status")
            continue
        c0 = int(evc[k]) - lat_samples
        # vecinos < 1.0 s (otro flanco del MISMO tramo), contando el propio flanco original
        same = np.isfinite(evc) & (seg_ev == seg_ev[k])
        d = np.abs(evc[same] - evc[k])
        if np.sum(d < OVERLAP_S * FS_IN) > 1 or (evt_flags[k] & 0x02):
            rs.append("overlap")
        cand = np.flatnonzero((counter == c0) & (seg_row == seg_ev[k]))
        if len(cand) == 0:
            rs.append("counter")
            continue
        i = int(cand[0]) if len(cand) == 1 or not np.isfinite(onset_arm[k]) else \
            int(cand[np.argmin(np.abs(t_arr[cand] - onset_arm[k]))])
        a, b = i - PRE_SAMPLES, i + POST_SAMPLES
        if a < 0 or b > n_rows or not np.all(np.diff(counter[a:b]) == 1):
            rs.append("counter")
            continue
        if np.any(flags[a:b] & F_REJECT):
            rs.append("flags")
        if np.any(flags[a:b] & F_UNFILTERED):
            rs.append("unfiltered")
        if has_raw and np.nanmax(np.abs(r[a:b])) >= ADC_SAT_UV:
            rs.append("saturated")
        win = f[a:b]                                       # [250, 8] uV, ya filtrado por el puente
        W[k] = win.T                                       # para C2: la preprocesa como el S3 (float32)
        win = win - win[:PRE_SAMPLES].mean(axis=0)         # baseline [-200, 0) ms
        x = win[PRE_SAMPLES:].reshape(N_T, DECIMATION, N_CH).mean(axis=1).T   # [8, 40] boxcar x5
        X[k] = x
        max_abs[k] = np.abs(x).max()
        if np.any(x.std(axis=1) < FLAT_STD_UV):
            rs.append("flat")
        if max_abs[k] > gate_uv:
            rs.append("amplitude")
        gyro_m[k] = gyro_metric(gyr[a:b])
        if gyro_m[k] > gate_gyro:
            rs.append("gyro")

    rejected = np.array([len(x) > 0 for x in reasons])
    reject_reason = np.array(["+".join(x) if x else "ok" for x in reasons])
    label = ev["label"].astype(str).to_numpy()
    y = np.where(label == "correct", 0, np.where(label == "error", 1, -1)).astype(np.int8)

    # calibracion: primeras n_cal acciones correctas y limpias, en orden de la sesion
    clean_correct = np.flatnonzero((y == 0) & ~rejected)
    cal_idx = clean_correct[:n_cal]
    is_cal = np.zeros(n, bool)
    is_cal[cal_idx] = True
    norm_valid = len(cal_idx) >= MIN_CAL
    if len(cal_idx):
        xc = X[cal_idx].astype(np.float64)
        mean = xc.mean(axis=(0, 2))
        std = np.maximum(xc.std(axis=(0, 2)), STD_FLOOR)
    else:
        mean, std = np.zeros(N_CH), np.ones(N_CH)
    if not norm_valid:
        print(f"  AVISO: solo {len(cal_idx)} epocas de calibracion (< {MIN_CAL}); norm_valid = False", file=sys.stderr)

    session = path.name
    subject = str(meta.get("sujeto", meta.get("subject", session.split("_", 2)[-1])))
    filt = meta.get("filtro", meta.get("filter", meta.get("filter_design", None)))
    if filt is None:
        filt = "scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos') + sosfilt causal"
    elif not isinstance(filt, str):
        filt = json.dumps(filt, ensure_ascii=False)

    out = dict(
        X=X,
        W=W,
        y=y,
        rejected=rejected,
        reject_reason=reject_reason,
        action_id=pd.to_numeric(ev["action_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
        evt_counter=np.where(np.isfinite(evc), evc, -1).astype(np.int64),
        status=status.astype(str),
        is_calibration=is_cal,
        max_abs_uv=max_abs,
        gyro_metric=gyro_m,
        norm_mean=mean.astype(np.float32),
        norm_std=std.astype(np.float32),
        norm_valid=np.bool_(norm_valid),
        has_raw=np.bool_(has_raw),
        n_calibration=np.int64(len(cal_idx)),
        ch_names=np.array(CH),
        subject=np.array(subject),
        session=np.array(session),
        fs_in=np.int64(FS_IN),
        fs_out=np.int64(FS_OUT),
        window_ms=np.array(WINDOW_MS, np.int64),
        baseline_ms=np.array(BASELINE_MS, np.int64),
        decimation=np.int64(DECIMATION),
        latency_offset_ms=np.float64(latency_ms),
        gate_uv=np.float64(gate_uv),
        gate_gyro=np.float64(gate_gyro),
        flat_std_uv=np.float64(FLAT_STD_UV),
        adc_sat_uv=np.float64(ADC_SAT_UV),
        filter=np.array(filt),
        pipeline_version=np.array(PIPELINE_VERSION),
    )
    return out


def summarize(out):
    n = len(out["y"])
    rej = out["rejected"]
    print(f"  {out['session']}: {n} acciones, {int((out['y'] == 0).sum())} correct, "
          f"{int((out['y'] == 1).sum())} error, {int((out['y'] == -1).sum())} manual; "
          f"rechazadas {int(rej.sum())} ({100 * rej.mean():.1f} %), calibracion {int(out['n_calibration'])}")
    prim = pd.Series([s.split("+")[0] for s in out["reject_reason"][rej]]).value_counts()
    for k, v in prim.items():
        print(f"      {k}: {v}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path, help="carpetas crudas (por defecto, todas en ml/data/raw/)")
    ap.add_argument("--out", type=Path, default=here / "processed")
    ap.add_argument("--latency-offset-ms", type=float, default=EVENT_LATENCY_OFFSET_MS)
    ap.add_argument("--gate-uv", type=float, default=GATE_UV)
    ap.add_argument("--gate-gyro", type=float, default=GATE_GYRO)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--suggest-gyro", action="store_true", help="solo imprime un GATE_GYRO sugerido")
    ap.add_argument("--export-c2", type=Path, metavar="NPZ",
                    help="además, juntar todas las sesiones en un .npz para ml/autoencoder/train_errp_ae.py")
    ap.add_argument("--c2-include-calibration", action="store_true",
                    help="no quitar las épocas de calibración del export para C2")
    a = ap.parse_args()

    sessions = a.sessions or sorted(p for p in (here / "raw").glob("*") if (p / "eeg.csv").exists())
    if not sessions:
        sys.exit("no hay sesiones crudas (ml/data/raw/*/eeg.csv); prueba ml/data/make_synthetic.py")
    a.out.mkdir(parents=True, exist_ok=True)
    built = []
    for s in sessions:
        if a.suggest_gyro:
            eeg, _, _, blocks = load_session(s)
            res = suggest_gyro(eeg, blocks) if blocks is not None else None
            print(s.name, res if res else "sin bloques eyes_open/head suficientes")
            continue
        out = build_session(s, a.latency_offset_ms, a.gate_uv, a.gate_gyro, a.n_cal)
        dest = a.out / f"{s.name}.npz"
        np.savez_compressed(dest, **out)
        summarize(out)
        print(f"  -> {dest}")
        built.append(out)
    if a.export_c2 and built:
        export_c2(built, a.export_c2, a.c2_include_calibration)


def export_c2(outs, dest: Path, include_calibration=False):
    """Formato de entrada de train_errp_ae.py (C2): windows [N, 8, 250] float32 (f_* tras el IIR,
    sin baseline; C2 aplica errp_ae.preprocess_window = ae_preprocess del S3), epochs [N, 8, 40]
    (X, referencia), is_error [N] bool, session [N] int64 (índice de sesión). Solo épocas limpias
    correct/error; sin las de calibración salvo include_calibration (CLAUDE.md: no se entrenan)."""
    win, ep, err, ses, names = [], [], [], [], []
    for i, o in enumerate(outs):
        keep = ~o["rejected"] & (o["y"] >= 0)
        if not include_calibration:
            keep &= ~o["is_calibration"]
        win.append(o["W"][keep])
        ep.append(o["X"][keep])
        err.append(o["y"][keep] == 1)
        ses.append(np.full(int(keep.sum()), i, np.int64))
        names.append(str(o["session"]))
    arrays = dict(windows=np.concatenate(win), epochs=np.concatenate(ep), is_error=np.concatenate(err),
                  session=np.concatenate(ses), session_names=np.array(names),
                  latency_offset_ms=np.float64(outs[0]["latency_offset_ms"]),
                  pipeline_version=np.array(PIPELINE_VERSION))
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **arrays)
    n, ne = len(arrays["is_error"]), int(arrays["is_error"].sum())
    print(f"export C2: {n} épocas limpias ({n - ne} correct, {ne} error) de {len(outs)} sesiones -> {dest}")


if __name__ == "__main__":
    main()
