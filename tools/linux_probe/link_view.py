#!/usr/bin/env python3
"""
Lee las tramas link_protocol.h que el puente copia al USB (Kconfig
BRIDGE_LINK_MIRROR_CONSOLE), las graba, las grafica y verifica el IIR.

    UV="uv run --with pyserial --with numpy --with scipy --with matplotlib"
    $UV tools/linux_probe/link_view.py live                 # señal + espectro en vivo
    $UV tools/linux_probe/link_view.py record -o s.csv -d 120
    $UV tools/linux_probe/link_view.py plot s.csv           # trazas, espectro, flags
    $UV tools/linux_probe/link_view.py check s.csv          # prueba 7: ESP32 vs scipy

CSV: t_llegada, counter, flags, 8 canales filtrados (f_*), acc (g), gyr (°/s),
8 crudos (r_*, si BRIDGE_LINK_RAW). r_* es exactamente lo que entró al filtro (incluye ZOH).
"""

import argparse
import csv
import glob
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

# ---- link_protocol.h ----
SYNC = b"\xA5\x5A"
T_EEG, T_STATUS, T_EEG_RAW = 0x01, 0x02, 0x03
F_HELD, F_GAP, F_SETTLING, F_RESET, F_SESSION, F_UNFILT = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
F_REJECT = F_HELD | F_GAP | F_SETTLING  # igual que LINK_F_REJECT
FLAG_NAMES = {F_HELD: "HELD", F_GAP: "GAP", F_SETTLING: "SETTLING", F_RESET: "RESET",
              F_SESSION: "SESSION", F_UNFILT: "UNFILT"}
EEG_FMT = "<IB8f3h3h"  # 49 B: counter, flags, 8 µV, acc[3], gyr[3]
EEG_LEN = 49
ACC_SCALE, GYR_SCALE = 1 / 4096.0, 1 / 32.8
STATUS_FMT = "<BBHIIII"  # 20 B
STATES = {0: "IDLE", 1: "CONNECTING", 2: "STREAMING"}
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
FS = 250.0
BAUD = 921600


def crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class LinkRx:
    """Igual que link_rx_t: busca A5 5A, valida len y CRC."""

    def __init__(self):
        self.buf = bytearray()
        self.frames = self.crc_errors = 0

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                del self.buf[:-1]
                return out
            del self.buf[:i]
            if len(self.buf) < 4:
                return out
            ln = self.buf[3]
            if ln > 64:
                del self.buf[:2]
                continue
            total = 4 + ln + 2
            if len(self.buf) < total:
                return out
            fr = bytes(self.buf[:total])
            if struct.unpack_from("<H", fr, total - 2)[0] == crc16(fr[2:4 + ln]):
                self.frames += 1
                out.append((fr[2], fr[4:4 + ln]))
                del self.buf[:total]
            else:
                self.crc_errors += 1
                del self.buf[:2]  # reintentar desde el siguiente byte


def flag_str(f):
    return "|".join(n for b, n in FLAG_NAMES.items() if f & b) or "-"


def open_port(port):
    import serial

    port = port or (sorted(glob.glob("/dev/ttyUSB*")) or [None])[0]
    if not port:
        sys.exit("no hay /dev/ttyUSB*")
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, BAUD, 0.05
    s.dtr = s.rts = False  # intentar no reiniciar la placa (el CH340 puede hacerlo igual)
    s.open()
    print(f"[link] {port} @ {BAUD}", file=sys.stderr)
    return s


def samples(port, duration=None):
    """Genera (t, kind, payload_tuple). kind: 'eeg' | 'raw' | 'status'."""
    s = open_port(port)
    rx = LinkRx()
    t_end = time.time() + duration if duration else None
    try:
        while not t_end or time.time() < t_end:
            data = s.read(4096)
            t = time.time()
            for typ, pl in rx.feed(data):
                if typ in (T_EEG, T_EEG_RAW) and len(pl) == EEG_LEN:
                    yield t, "eeg" if typ == T_EEG else "raw", struct.unpack(EEG_FMT, pl)
                elif typ == T_STATUS and len(pl) == 20:
                    yield t, "status", struct.unpack(STATUS_FMT, pl) + (rx.crc_errors,)
    finally:
        s.close()


def print_status(st):
    state, bat, proc, frames, gaps, lost, reconn, crc = st
    loss = 100.0 * lost / (frames + lost) if frames + lost else 0.0
    bat_s = "?" if bat == 0xFF else f"{bat}%"
    print(f"[status] {STATES.get(state, state)} bat={bat_s} tramas={frames} huecos={gaps} "
          f"perdidas={lost} ({loss:.3f}%) reconex={reconn} proc_max={proc} us crc_pc={crc}", file=sys.stderr)


def cmd_record(a):
    out = open(a.out, "w", newline="")
    w = csv.writer(out)
    imu_cols = [f"acc_{a}" for a in "xyz"] + [f"gyr_{a}" for a in "xyz"]
    w.writerow(["t", "counter", "flags"] + [f"f_{c}" for c in CH] + imu_cols + [f"r_{c}" for c in CH])
    pending = None  # fila EEG esperando su EEG_RAW (llega justo detrás)
    n = 0

    def flush():
        nonlocal pending, n
        if pending:
            w.writerow(pending + [""] * 8)
            n += 1
            pending = None

    try:
        for t, kind, p in samples(a.port, a.duration):
            if kind == "eeg":
                flush()
                imu = [f"{v * ACC_SCALE:.4f}" for v in p[10:13]] + [f"{v * GYR_SCALE:.3f}" for v in p[13:16]]
                pending = [f"{t:.4f}", p[0], p[1]] + [f"{v:.4f}" for v in p[2:10]] + imu
            elif kind == "raw" and pending and pending[1] == p[0]:
                w.writerow(pending + [f"{v:.4f}" for v in p[2:10]])
                n += 1
                pending = None
            elif kind == "status":
                print_status(p)
    except KeyboardInterrupt:
        pass
    flush()
    out.close()
    print(f"[record] {n} muestras -> {a.out}", file=sys.stderr)


def load(path):
    import numpy as np

    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = [row for row in r]
    cnt = np.array([int(x[1]) for x in rows])
    flags = np.array([int(x[2]) for x in rows])
    filt = np.array([[float(v) for v in x[3:11]] for x in rows])
    has_raw = all(x[17] != "" for x in rows) if rows else False
    raw = np.array([[float(v) for v in x[17:25]] for x in rows]) if has_raw else None
    return cnt, flags, filt, raw


def psd(x):
    from scipy import signal

    return signal.welch(x, fs=FS, nperseg=min(len(x), 512), axis=0)


def cmd_plot(a):
    import matplotlib.pyplot as plt
    import numpy as np

    cnt, flags, filt, raw = load(a.csv)
    t = (cnt - cnt[0]) / FS  # tiempo por contador, no por llegada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [3, 1]})
    sp = a.spacing
    for i, c in enumerate(CH):
        ax1.plot(t, filt[:, i] - i * sp, lw=0.6, label=c)
    ax1.set_yticks([-i * sp for i in range(8)], CH)
    ax1.set_xlabel("s (contador × 4 ms)")
    ax1.set_title(f"Filtrado 1–15 Hz (separación {sp:g} µV)")
    for mask, color, name in ((flags & F_REJECT, "red", "GAP/SETTLING"), (flags & F_HELD, "orange", "HELD")):
        idx = np.flatnonzero(mask)
        if idx.size:
            ax1.scatter(t[idx], np.full(idx.size, sp * 0.8), s=4, c=color, label=name)
    ax1.legend(loc="upper right", fontsize=7)

    ok = (flags & F_REJECT) == 0
    sig = raw if (a.raw and raw is not None) else filt
    f, p = psd(sig[ok] - sig[ok].mean(axis=0))
    for i, c in enumerate(CH):
        ax2.semilogy(f, p[:, i], lw=0.8, label=c)
    ax2.axvspan(8, 12, color="0.9")
    ax2.set_xlim(0, 40 if a.raw else 20)
    ax2.set_xlabel("Hz")
    ax2.set_title("PSD (Welch)" + (" crudo" if a.raw else " filtrado") + "; gris = alfa")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


def cmd_check(a):
    """Prueba 7: reproducir el IIR del ESP32 con scipy sobre r_* y comparar con f_*."""
    import numpy as np
    from scipy import signal

    sys.path.insert(0, str(Path(__file__).parent))
    from design_iir import design_sos

    cnt, flags, filt, raw = load(a.csv)
    if raw is None:
        sys.exit("sin columnas r_* (activar BRIDGE_LINK_RAW)")
    if flags[0] & F_UNFILT:
        sys.exit("grabado con BRIDGE_UNFILTERED: no hay filtro que comparar")
    sos = design_sos()
    zi1 = signal.sosfilt_zi(sos)
    starts = list(np.flatnonzero(flags & F_RESET))
    if not starts or starts[0] != 0:
        print("aviso: la grabación no empieza en un reinicio del filtro; se descarta hasta el primero")
    if not starts:
        sys.exit("no hay ningún FILTER_RESET en la grabación")
    starts.append(len(cnt))
    worst = 0.0
    for s0, s1 in zip(starts[:-1], starts[1:]):
        x = raw[s0:s1]
        y = np.empty_like(x)
        for ch in range(8):
            y[:, ch], _ = signal.sosfilt(sos, x[:, ch], zi=zi1 * x[0, ch])
        err = np.abs(y - filt[s0:s1]).max(axis=0)
        worst = max(worst, err.max())
        print(f"segmento cnt {cnt[s0]}..{cnt[s1 - 1]} ({s1 - s0} muestras): error máx por canal µV = "
              + " ".join(f"{c}={e:.4f}" for c, e in zip(CH, err)))
    held = int((flags & F_HELD).astype(bool).sum())
    print(f"muestras={len(cnt)} HELD={held} GAP={int((flags & F_GAP).astype(bool).sum())} "
          f"reinicios={len(starts) - 1}")
    verdict = "PASA" if worst < 0.01 else "NO PASA"
    print(f"prueba 7: error máx {worst:.5f} µV (criterio < 0.01 µV) -> {verdict}")


def cmd_live(a):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    n = int(a.window * FS)
    buf = deque(maxlen=n)
    flagbuf = deque(maxlen=n)
    status = ["esperando datos..."]

    def reader():
        for _, kind, p in samples(a.port):
            if kind == "eeg":
                buf.append(p[2:10])
                flagbuf.append(p[1])
            elif kind == "status":
                st = p
                bat = "?" if st[1] == 0xFF else f"{st[1]}%"
                status[0] = (f"{STATES.get(st[0], st[0])}  bat={bat}  tramas={st[3]}  huecos={st[4]}  "
                             f"perdidas={st[5]}  proc_max={st[2]} us  crc_pc={st[7]}")

    threading.Thread(target=reader, daemon=True).start()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [3, 1]})
    sp = a.spacing
    tt = np.arange(n) / FS - a.window
    lines = [ax1.plot(tt, np.full(n, np.nan), lw=0.6)[0] for _ in CH]
    ax1.set_yticks([-i * sp for i in range(8)], CH)
    ax1.set_ylim(-8 * sp, sp)
    ax1.set_xlabel("s")
    plines = [ax2.semilogy([], [], lw=0.8, label=c)[0] for c in CH]
    ax2.axvspan(8, 12, color="0.9")
    ax2.set_xlim(0, 20)
    ax2.set_ylim(1e-3, 1e4)
    ax2.set_xlabel("Hz")
    ax2.set_title("PSD últimos 4 s; gris = alfa")
    ax2.legend(fontsize=7)
    title = fig.suptitle("")

    def update(_):
        title.set_text(status[0])
        if len(buf) < 50:
            return
        x = np.array(buf)
        k = len(x)
        for i, ln in enumerate(lines):
            y = np.full(n, np.nan)
            y[-k:] = x[:, i] - i * sp
            ln.set_ydata(y)
        seg = x[-int(4 * FS):]
        seg_flags = np.array(flagbuf)[-len(seg):]
        seg = seg[(seg_flags & F_REJECT) == 0]
        if len(seg) > 256:
            f, p = psd(seg - seg.mean(axis=0))
            for i, ln in enumerate(plines):
                ln.set_data(f, p[:, i])

    _anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="por defecto el primer /dev/ttyUSB*")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("-o", "--out", default="link.csv")
    r.add_argument("-d", "--duration", type=float, help="segundos (Ctrl+C también corta)")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--spacing", type=float, default=50.0, help="µV entre canales")
    p.add_argument("--raw", action="store_true", help="espectro de r_* (sin filtrar)")
    c = sub.add_parser("check")
    c.add_argument("csv")
    lv = sub.add_parser("live")
    lv.add_argument("--window", type=float, default=5.0, help="segundos visibles")
    lv.add_argument("--spacing", type=float, default=50.0, help="µV entre canales")
    a = ap.parse_args()
    {"record": cmd_record, "plot": cmd_plot, "check": cmd_check, "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
