#!/usr/bin/env python3
"""
INTUNE: grabación del paradigma ErrP con el brazo real (C4; escrito por C1).

El operador observa el brazo. La PC ordena cada acción; ~22 % son errores deliberados
impredecibles. El brazo dispara el pulso de sincronía (GPIO) al inicio de la acción; ese
cable llega al puente (BRIDGE_SYNC_GPIO = GPIO18, tierra común) y el puente envía una
trama EVENT con el número de muestra Unicorn. La etiqueta la pone la PC, que sabe qué
ordenó; el tiempo lo pone el flanco (regla dura 4).

    uv run --with pyserial --with numpy --with scipy tools/recording/record_errp.py \\
        --subject S01 --bridge-port /dev/ttyUSB0 --arm-port /dev/ttyUSB1

    --no-arm      prueba de banco sin brazo: la PC anuncia cada acción y espera un
                  flanco manual en GPIO18 (cable a 3.3 V); etiqueta "manual".
    --no-blocks   saltar los bloques de calibración/artefactos.

Protocolo PC <-> ESP32 del brazo (C3), texto por USB a 115200, líneas con \\n:
    PC  -> brazo: ACT <id> <err>     err = 0 acción correcta, 1 error deliberado
    brazo -> PC : ONSET <id>         justo después de subir el pulso GPIO (>= 5 ms en alto)
    brazo -> PC : DONE <id>          acción terminada, listo para la siguiente
    brazo -> PC : ERR <id> <texto>   no pudo ejecutar la acción
Acciones de ~2 s; la PC respeta >= 1.5 s entre inicios (CLAUDE.md).

Salida: ml/data/raw/<fecha>_<sujeto>/
    eeg.csv     igual que link_view.py record (contador, flags, f_*, acc, gyr, r_*)
    events.csv  una fila por acción: etiqueta, flanco (seq, counter, offset_us, flags), estado
    blocks.csv  bloques de calibración: nombre, contador inicial y final
    status.csv  STATUS del puente cada segundo (pérdidas, reconexiones, batería)
    meta.json   sujeto, parámetros, diseño del filtro
"""

import argparse
import csv
import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "linux_probe"))
import link_view as L  # noqa: E402

BLOCKS = [
    ("eyes_open", 60, "Ojos ABIERTOS, mirando un punto fijo, relajado."),
    ("eyes_closed", 60, "Ojos CERRADOS, relajado."),
    ("blinks", 20, "Parpadea fuerte cada ~2 s."),
    ("jaw", 20, "Aprieta la mandíbula 1 s cada ~3 s."),
    ("head", 20, "Mueve la cabeza (sí/no) despacio."),
]


def make_schedule(n, p_error, rng):
    """Etiquetas 0/1 con round(n*p) errores, impredecibles: primeras 5 correctas y
    nunca más de 2 errores seguidos."""
    n_err = round(n * p_error)
    while True:
        lab = [1] * n_err + [0] * (n - n_err)
        rng.shuffle(lab)
        if any(lab[:5]):
            continue
        if any(lab[i] and lab[i + 1] and lab[i + 2] for i in range(n - 2)):
            continue
        return lab


class Bridge(threading.Thread):
    """Lee el stream del puente, escribe eeg.csv y status.csv, y pasa los EVENT por cola."""

    def __init__(self, port, outdir):
        super().__init__(daemon=True)
        self.port = port
        self.events = queue.Queue()
        self.last_counter = 0
        self.state = None
        self.settled_since = None  # contador desde el que no hay SETTLING
        self.n = 0
        self.last_status = None
        self.reboots = 0
        self._eeg = open(outdir / "eeg.csv", "w", newline="")
        self._st = open(outdir / "status.csv", "w", newline="")
        self.w = csv.writer(self._eeg)
        self.ws = csv.writer(self._st)
        imu = [f"acc_{a}" for a in "xyz"] + [f"gyr_{a}" for a in "xyz"]
        self.w.writerow(["t", "counter", "flags"] + [f"f_{c}" for c in L.CH] + imu + [f"r_{c}" for c in L.CH])
        self.ws.writerow(["t", "state", "battery", "proc_us_max", "frames", "gaps", "lost", "reconnects", "crc_pc"])
        self.stop = False

    def run(self):
        pending = None
        for t, kind, p in L.samples(self.port):
            if self.stop:
                break
            if kind == "eeg":
                if pending:
                    self.w.writerow(pending + [""] * 8)
                imu = [f"{v * L.ACC_SCALE:.4f}" for v in p[10:13]] + [f"{v * L.GYR_SCALE:.3f}" for v in p[13:16]]
                pending = [f"{t:.4f}", p[0], p[1]] + [f"{v:.4f}" for v in p[2:10]] + imu
                self.last_counter = p[0]
                self.n += 1
                if p[1] & L.F_SETTLING:
                    self.settled_since = None
                elif self.settled_since is None:
                    self.settled_since = p[0]
            elif kind == "raw" and pending and pending[1] == p[0]:
                self.w.writerow(pending + [f"{v:.4f}" for v in p[2:10]])
                pending = None
            elif kind == "event":
                self.events.put((t, p))
            elif kind == "status":
                if self.last_status and p[3] < self.last_status[3]:
                    self.reboots += 1  # contadores a cero: el puente se reinició
                    print("\n!! el puente se reinició (contadores a cero)", file=sys.stderr)
                self.last_status = p
                self.state = L.STATES.get(p[0], p[0])
                self.ws.writerow([f"{t:.3f}", self.state, p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
                self._st.flush()
        self._eeg.close()
        self._st.close()

    def settled_s(self):
        if self.settled_since is None:
            return 0.0
        return (self.last_counter - self.settled_since) / L.FS


class Arm:
    def __init__(self, port):
        import serial

        self.s = serial.Serial(port, 115200, timeout=0.05)
        time.sleep(2.0)  # abrir el CH340 reinicia la placa del brazo
        self.s.reset_input_buffer()
        self.buf = b""

    def send(self, line):
        self.s.write((line + "\n").encode())

    def lines(self):
        self.buf += self.s.read(256)
        out = []
        while b"\n" in self.buf:
            ln, self.buf = self.buf.split(b"\n", 1)
            ln = ln.decode(errors="replace").strip()
            if ln:
                out.append((time.time(), ln))
        return out


def wait_ready(br):
    print("Esperando al puente (STREAMING y 3 s sin SETTLING)...")
    while not (br.state == "STREAMING" and br.settled_s() >= 3.0):
        st = br.last_status
        msg = f"  estado={br.state} muestras={br.n}"
        if st:
            msg += f" bat={st[1]}% perdidas={st[5]}"
        print(msg, end="\r")
        time.sleep(0.5)
    print("\nPuente listo.")


def run_blocks(br, wb):
    for name, dur, instr in BLOCKS:
        input(f"\n[bloque {name}, {dur} s] {instr}\n  Enter para empezar (Ctrl+C aborta)... ")
        c0 = br.last_counter
        t_end = time.time() + dur
        while time.time() < t_end:
            print(f"  {name}: {t_end - time.time():4.0f} s", end="\r")
            time.sleep(0.2)
        c1 = br.last_counter
        wb.writerow([name, c0, c1])
        print(f"  {name}: listo (cnt {c0}..{c1})          ")


def run_actions(a, br, arm, labels, we, rng):
    n = len(labels)
    stats = {"ok": 0, "no_pulse": 0, "no_t0": 0, "overlap": 0, "arm_err": 0}
    last_seq = None
    for i, err in enumerate(labels, 1):
        if i > 1 and (i - 1) % a.rest_every == 0:
            input(f"\nDescanso ({i - 1}/{n}). Enter para seguir... ")
        # vaciar eventos sueltos (rebotes, pulsos fuera de acción)
        while not br.events.empty():
            br.events.get_nowait()
        t_sent = time.time()
        label = "manual" if arm is None else ("error" if err else "correct")
        if arm is None:
            print(f"\nACCIÓN {i}/{n}: da un pulso en GPIO18 ahora", end="", flush=True)
        else:
            arm.send(f"ACT {i} {err}")
        t_onset = t_done = None
        evt = None
        arm_msg = ""
        deadline = t_sent + a.pulse_timeout
        while time.time() < deadline and (evt is None or (arm is not None and t_done is None)):
            try:
                t_evt, evt_p = br.events.get(timeout=0.02)
                if evt is None:
                    evt = (t_evt, evt_p)
            except queue.Empty:
                pass
            if arm is not None:
                for t_ln, ln in arm.lines():
                    parts = ln.split()
                    if len(parts) >= 2 and parts[1] == str(i):
                        if parts[0] == "ONSET":
                            t_onset = t_ln
                        elif parts[0] == "DONE":
                            t_done = t_ln
                        elif parts[0] == "ERR":
                            arm_msg = ln
                            t_done = t_ln
            if evt is not None and arm is not None and t_done is None:
                deadline = max(deadline, evt[0] + a.action_timeout)
        if arm_msg:
            status = "arm_err"
        elif evt is None:
            status = "no_pulse"
        elif evt[1][3] & L.EVT_NO_T0:
            status = "no_t0"
        elif evt[1][3] & L.EVT_OVERLAP:
            status = "overlap"
        else:
            status = "ok"
        stats[status] += 1
        seq, cnt, off, fl = evt[1] if evt else ("", "", "", "")
        if evt and last_seq is not None and seq != last_seq + 1:
            print(f"\n!! saltaron flancos: seq {last_seq} -> {seq}", file=sys.stderr)
        if evt:
            last_seq = seq
        we.writerow([i, label, f"{t_sent:.4f}", f"{t_onset:.4f}" if t_onset else "",
                     f"{t_done:.4f}" if t_done else "", seq, cnt, off, fl, status, arm_msg])
        print(f"\r[{i:3d}/{n}] {label:7s} {status:8s} cnt={cnt}   "
              + " ".join(f"{k}={v}" for k, v in stats.items()), end="", flush=True)
        # >= isi entre inicios (desde el flanco si lo hubo)
        t_ref = evt[0] if evt else t_sent
        wait = rng.uniform(*a.isi) - (time.time() - t_ref)
        if wait > 0:
            time.sleep(wait)
    print()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--bridge-port", help="por defecto el primer /dev/ttyUSB*")
    ap.add_argument("--arm-port")
    ap.add_argument("--no-arm", action="store_true", help="banco: pulsos manuales, sin brazo")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--n-actions", type=int, default=300)
    ap.add_argument("--p-error", type=float, default=0.22)
    ap.add_argument("--isi", type=float, nargs=2, default=(2.5, 3.5), metavar=("MIN", "MAX"),
                    help="s entre inicios de acción (>= 1.5 s y >= duración de la acción)")
    ap.add_argument("--rest-every", type=int, default=50)
    ap.add_argument("--pulse-timeout", type=float, default=3.0, help="s para recibir el flanco")
    ap.add_argument("--action-timeout", type=float, default=6.0, help="s tras el flanco para DONE")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default=str(ROOT / "ml" / "data" / "raw"))
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.isi[0] < 1.5:
        sys.exit("--isi mínimo 1.5 s (CLAUDE.md)")
    if not a.no_arm and not a.arm_port:
        sys.exit("falta --arm-port (o --no-arm para la prueba de banco)")

    seed = a.seed if a.seed is not None else int(time.time())
    rng = random.Random(seed)
    labels = make_schedule(a.n_actions, a.p_error, rng)
    outdir = Path(a.out) / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.subject}"
    outdir.mkdir(parents=True)

    from design_iir import BAND, FS, N
    meta = {
        "subject": a.subject, "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
        "n_actions": a.n_actions, "p_error": a.p_error, "isi_s": list(a.isi), "no_arm": a.no_arm,
        "filter": f"scipy.signal.butter({N}, {list(BAND)}, btype='band', fs={FS:g}, output='sos'), causal",
        "channels": L.CH, "fs": L.FS, "sync": "flanco GPIO del brazo -> puente GPIO18 -> trama EVENT",
        "notes": a.notes,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Sesión: {outdir}")

    br = Bridge(a.bridge_port, outdir)
    br.start()
    arm = None if a.no_arm else Arm(a.arm_port)
    stats = {}
    try:
        wait_ready(br)
        if not a.no_blocks:
            with open(outdir / "blocks.csv", "w", newline="") as fb:
                wb = csv.writer(fb)
                wb.writerow(["block", "counter_start", "counter_end"])
                run_blocks(br, wb)
        input(f"\nParadigma: {a.n_actions} acciones, ~{a.p_error:.0%} errores. Observa el brazo. "
              "Enter para empezar... ")
        with open(outdir / "events.csv", "w", newline="") as fe:
            we = csv.writer(fe)
            we.writerow(["action_id", "label", "t_sent", "t_onset_arm", "t_done_arm",
                         "evt_seq", "evt_counter", "evt_offset_us", "evt_flags", "status", "arm_msg"])
            stats = run_actions(a, br, arm, labels, we, rng)
    except KeyboardInterrupt:
        print("\nInterrumpido; se guarda lo grabado.")
    finally:
        time.sleep(1.0)  # que entren las últimas muestras de las épocas finales
        br.stop = True
        st = br.last_status
        meta.update({"end": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": br.n, "bridge_reboots": br.reboots,
                     "action_stats": stats,
                     "bridge_final": dict(zip(["state", "battery", "proc_us_max", "frames", "gaps", "lost",
                                               "reconnects", "crc_pc"], st)) if st else None})
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"Guardado en {outdir}  muestras={br.n}  reinicios_puente={br.reboots}  {stats}")


if __name__ == "__main__":
    main()
