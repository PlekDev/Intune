#!/usr/bin/env python3
"""
INTUNE: visor en vivo de una grabación (C4). Escucha por UDP lo que publica record_errp.py
(--ui-port, def. 47474) y muestra:
  - izquierda: los 8 canales filtrados (últimos --window s), con una línea por acción:
      verde = correct, rojo = error, naranja = recovery, gris = check, azul = familiar;
  - derecha arriba: ruido robusto por canal (últimos 5 s) contra el umbral de la prueba de señal;
  - derecha abajo: promedio acumulado error vs correct en Fz y Cz (épocas [-200, +800] ms, línea base
    [-200, 0], mismas reglas de rechazo que el dataset: HELD/GAP/SETTLING o |x| > 100 µV).
El título dice la fase, la última acción y el estado del puente.

    .venv/bin/python tools/recording/live_view.py            # en otra terminal, antes o durante la sesión
run_session_v3.sh lo abre solo. Cerrar la ventana no afecta a la grabación.
"""

import argparse
import collections
import json
import socket
import threading
import time

import numpy as np

CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
FS = 250
PRE, POST = 50, 200                 # -200 ms, +800 ms
REJECT_FLAGS = 0x01 | 0x02 | 0x04   # HELD, GAP, SETTLING
GATE_UV = 100.0
COLORS = {"correct": "tab:green", "error": "tab:red", "recovery": "tab:orange", "check": "0.6",
          "familiar": "tab:blue", "manual": "tab:purple"}


class State:
    def __init__(self, keep_s):
        self.lock = threading.Lock()
        n = keep_s * FS
        self.cnt = collections.deque(maxlen=n)
        self.flags = collections.deque(maxlen=n)
        self.x = collections.deque(maxlen=n)
        self.acts = collections.deque(maxlen=200)       # (counter, label, i)
        self.pending = []                                # (counter, label, t_recv)
        self.sums = {"correct": np.zeros((POST + PRE, 8)), "error": np.zeros((POST + PRE, 8))}
        self.n = {"correct": 0, "error": 0}
        self.rejected = 0
        self.status = {}
        self.phase = "esperando a record_errp.py..."
        self.last_act = None
        self.check = None
        self.t_last = 0.0


def receiver(st, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        data, _ = sock.recvfrom(65535)
        try:
            m = json.loads(data)
        except ValueError:
            continue
        with st.lock:
            st.t_last = time.time()
            k = m.get("k")
            if k == "eeg":
                for row in m["s"]:
                    if st.cnt and row[0] < st.cnt[-1]:   # contador reiniciado (reconexión): empezar de cero
                        st.cnt.clear(); st.flags.clear(); st.x.clear(); st.pending.clear(); st.acts.clear()
                    st.cnt.append(row[0]); st.flags.append(row[1]); st.x.append(row[2:10])
            elif k == "status":
                st.status = m
            elif k == "phase":
                st.phase = m["text"]
            elif k == "check":
                st.check = m
            elif k == "act":
                st.last_act = m
                if m.get("counter") is not None:
                    st.acts.append((m["counter"], m["label"], m["i"]))
                    if m["label"] in st.sums and m["status"] == "ok":
                        st.pending.append((m["counter"], m["label"], time.time()))


def cut_epochs(st):
    """Corta las épocas pendientes cuya ventana ya llegó entera."""
    if not st.cnt:
        return
    first, last = st.cnt[0], st.cnt[-1]
    keep = []
    for c, label, t in st.pending:
        if last < c + POST:
            if time.time() - t < 10:
                keep.append((c, label, t))
            continue
        i0 = c - PRE - first
        if i0 < 0 or i0 + PRE + POST > len(st.cnt) or st.cnt[i0] != c - PRE or st.cnt[i0 + PRE + POST - 1] != c + POST - 1:
            st.rejected += 1   # hueco largo en la ventana
            continue
        fl = np.array([st.flags[j] for j in range(i0, i0 + PRE + POST)])
        ep = np.array([st.x[j] for j in range(i0, i0 + PRE + POST)], dtype=float)
        ep -= ep[:PRE].mean(axis=0)
        if (fl & REJECT_FLAGS).any() or np.abs(ep).max() > GATE_UV:
            st.rejected += 1
            continue
        st.sums[label] += ep
        st.n[label] += 1
    st.pending = keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=47474)
    ap.add_argument("--window", type=float, default=6.0, help="segundos visibles")
    ap.add_argument("--spacing", type=float, default=60.0, help="µV entre canales")
    ap.add_argument("--max-uv", type=float, default=20.0, help="umbral de ruido de la prueba de señal")
    a = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    st = State(keep_s=40)
    threading.Thread(target=receiver, args=(st, a.port), daemon=True).start()

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("INTUNE: grabación en vivo")
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
    ax_sig = fig.add_subplot(gs[:, 0])
    ax_noise = fig.add_subplot(gs[0, 1])
    ax_fz = fig.add_subplot(gs[1, 1])
    ax_cz = fig.add_subplot(gs[2, 1], sharex=ax_fz, sharey=ax_fz)

    nwin = int(a.window * FS)
    tt = (np.arange(nwin) - nwin + 1) / FS
    sp = a.spacing
    lines = [ax_sig.plot(tt, np.full(nwin, np.nan), lw=0.6, color="k")[0] for _ in CH]
    ax_sig.set_yticks([-i * sp for i in range(8)], CH)
    ax_sig.set_ylim(-8 * sp, sp)
    ax_sig.set_xlim(tt[0], 0)
    ax_sig.set_xlabel("s (por contador Unicorn)")
    ax_sig.set_title(f"EEG filtrado 1-15 Hz (separación {sp:g} µV)")
    vlines = []

    bars = ax_noise.bar(CH, np.zeros(8), color="tab:green")
    ax_noise.axhline(a.max_uv, color="tab:red", ls="--", lw=1)
    ax_noise.set_ylim(0, 2 * a.max_uv)
    ax_noise.set_ylabel("µV")
    ax_noise.set_title("ruido robusto (últimos 5 s)", fontsize=9)

    te = (np.arange(PRE + POST) - PRE) * 1000 / FS
    avg = {}
    for ax, ch in ((ax_fz, 0), (ax_cz, 2)):
        for label, col in (("correct", "tab:green"), ("error", "tab:red")):
            avg[(ch, label)] = ax.plot(te, np.full(len(te), np.nan), color=col, lw=1.2)[0]
        avg[(ch, "diff")] = ax.plot(te, np.full(len(te), np.nan), color="tab:blue", lw=1.6)[0]
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(f"{CH[ch]} µV")
    ax_fz.set_ylim(-12, 12)
    ax_fz.set_xlim(te[0], te[-1])  # las líneas nacen vacías (NaN): sin esto el eje X no se escala
    ax_cz.set_xlabel("ms desde la acción")
    leg_txt = ax_fz.text(0.01, 0.97, "", transform=ax_fz.transAxes, va="top", fontsize=8)
    title = fig.suptitle("", fontsize=10)

    def update(_):
        with st.lock:
            cut_epochs(st)
            if st.cnt:
                x = np.array(st.x)[-nwin:]
                c = np.array(st.cnt)[-nwin:]
                k = len(x)
                for i, ln in enumerate(lines):
                    y = np.full(nwin, np.nan)
                    y[-k:] = x[:, i] - i * sp
                    ln.set_ydata(y)
                for v in vlines:
                    v.remove()
                vlines.clear()
                for cc, label, _ in st.acts:
                    tx = (cc - c[-1]) / FS
                    if tt[0] <= tx <= 0:
                        vlines.append(ax_sig.axvline(tx, color=COLORS.get(label, "k"), lw=1.2, alpha=0.8))
                seg = x[-5 * FS:]
                if len(seg) > FS:
                    r = 1.4826 * np.median(np.abs(seg - np.median(seg, axis=0)), axis=0)
                    for b, v in zip(bars, r):
                        b.set_height(min(v, 2 * a.max_uv))
                        b.set_color("tab:red" if v > a.max_uv else "tab:green")
            for ch in (0, 2):
                m = {}
                for label in ("correct", "error"):
                    if st.n[label]:
                        m[label] = st.sums[label][:, ch] / st.n[label]
                        avg[(ch, label)].set_ydata(m[label])
                if len(m) == 2:
                    avg[(ch, "diff")].set_ydata(m["error"] - m["correct"])
            leg_txt.set_text(f"correct n={st.n['correct']}  error n={st.n['error']}  rechazadas {st.rejected}\n"
                             "verde correct, rojo error, azul error - correct")
            s = st.status
            stat = (f"{s.get('state', '?')}  bat {s.get('bat', '?')}%  huecos {s.get('gaps', '?')}  "
                    f"perdidas {s.get('lost', '?')}  reconex {s.get('reconn', '?')}") if s else "sin estado del puente"
            la = st.last_act
            act = f"acción {la['i']}/{la['n']}: {la['label']} -> {la['target']} [{la['status']}]" if la else ""
            chk = ""
            if st.check:
                chk = f"  |  prueba bloque {st.check['block']}: {'BIEN' if st.check['passed'] else 'MAL'}"
            stale = "  |  SIN DATOS" if time.time() - st.t_last > 3 else ""
            title.set_text(f"{st.phase}  |  {act}{chk}\n{stat}{stale}")

    _anim = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    main()
