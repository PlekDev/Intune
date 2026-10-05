#!/usr/bin/env python3
"""
INTUNE: grabación del paradigma ErrP con el brazo real (C4; escrito por C1).

El operador observa el brazo. La PC ordena cada acción; ~22 % son errores deliberados
impredecibles. El supervisor del brazo (C3) dispara el pulso de sincronía (GPIO25, alto 1 ms)
al inicio de la acción; ese cable llega al puente (BRIDGE_SYNC_GPIO = GPIO18, GND común) y el
puente envía una trama EVENT con el número de muestra Unicorn. La etiqueta la pone la PC, que
sabe qué ordenó; el tiempo lo pone el flanco (regla dura 4).

Paradigma (v2): el brazo recorre un anillo de 4 posturas (RING: izquierda -> adelante -> derecha ->
atrás). Correcta = siguiente postura; error = postura aleatoria a >= 45° del destino correcto
(recorrido 35-60°); la siguiente correcta va al destino que se saltó.

    .venv/bin/python tools/recording/record_errp.py --subject S01 \
        --bridge-port /dev/serial/by-id/<puente> --arm-port /dev/serial/by-id/<supervisor>

    --no-arm      prueba de banco sin brazo: la PC anuncia cada acción y espera un
                  flanco manual en GPIO18 (cable a 3.3 V); etiqueta "manual".
    --no-blocks   saltar los bloques de calibración/artefactos.
    --vel         °/s del brazo (def. 17.5 -> ~2 s por acción).

Supervisor (firmware/arm_esp32, consola de texto a 115200 por USB; clase Sup de
firmware/arm_esp32/tools/move_ab.py): al abrir se reinicia; "s3 0" simula el heartbeat del S3
(sin él el brazo no se mueve), "vel", "home"; cada acción es "act <b> <s> <e> <h> <0|1>" en rad.
Inicio = línea "actions: PULSO acción"; fin = "motion: MOVING -> IDLE"; fallo = "... rechazado"
o "SIN pulso". Al terminar: "stop" y "s3 off" (el brazo queda quieto con torque).
v3.1: el reposo se mide desde que el brazo se DETIENE (--rest, def. 1.5-2.5 s) y siempre hay >= 1.5 s
entre inicios (CLAUDE.md). La primera correcta tras un error (el regreso al punto saltado) se etiqueta
"recovery" y no entra al dataset. Con --check-every hay un descanso (Enter) al final de cada bloque.

Salida: ml/data/raw/<fecha>_<sujeto>/
    eeg.csv     igual que link_view.py record (contador, flags, f_*, acc, gyr, r_*)
    events.csv  una fila por acción: etiqueta, flanco (seq, counter, offset_us, flags), estado
    blocks.csv  bloques de calibración: nombre, contador inicial y final
    status.csv  STATUS del puente cada segundo (pérdidas, reconexiones, batería)
    meta.json   sujeto, parámetros, diseño del filtro
"""

import argparse
import collections
import contextlib
import csv
import json
import math
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


class Publisher:
    """Copia por UDP a localhost lo que pasa en la grabación, para tools/recording/live_view.py.
    sendto no bloquea y los errores se ignoran: si no hay visor, no cambia nada."""

    def __init__(self, port):
        import socket

        self.addr = ("127.0.0.1", port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if port else None
        self.buf = []

    def send(self, msg):
        if self.sock:
            with contextlib.suppress(OSError):
                self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)

    def sample(self, counter, flags, f):
        if self.sock:
            self.buf.append([counter, flags] + [round(v, 2) for v in f])
            if len(self.buf) >= 20:  # ~80 ms por paquete
                self.send({"k": "eeg", "s": self.buf})
                self.buf = []


PUB = Publisher(0)  # se reemplaza en main() según --ui-port


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
        self.recent = collections.deque(maxlen=int(L.FS * 240))  # (t, flags, f[8]) de los últimos 4 min

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
                self.recent.append((t, p[1], p[2:10]))
                PUB.sample(p[0], p[1], p[2:10])
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
                PUB.send({"k": "status", "state": self.state, "bat": p[1], "frames": p[3], "gaps": p[4],
                          "lost": p[5], "reconn": p[6], "crc": p[7]})
                self.ws.writerow([f"{t:.3f}", self.state, p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
                self._st.flush()
        self._eeg.close()
        self._st.close()

    def settled_s(self):
        if self.settled_since is None:
            return 0.0
        return (self.last_counter - self.settled_since) / L.FS


# Paradigma con el brazo real: anillo de 4 posturas (grados: base, hombro, codo, pinza) que el
# operador ve en orden IZQUIERDA -> ADELANTE -> DERECHA -> ATRÁS -> IZQUIERDA ... La acción correcta
# va a la SIGUIENTE postura del anillo. El error deliberado (v2) va a una postura ALEATORIA que se
# aparta >= ERR_MIN_DEV_DEG del destino correcto en alguna articulación, para que se note desde el
# arranque y no tarde en reconocerse (en la sesión S01 con "ir al revés" el error tardaba en verse).
# La siguiente acción correcta va al destino que se saltó y el anillo sigue desde ahí.
# El recorrido del error se limita a ERR_MOVE_DEG para que dure parecido a una correcta (35°):
# si fuera mucho más largo, el modelo podría aprender "movimiento grande" en vez de ErrP.
RING = [
    ("izquierda", (-35.0, 0.0, 90.0, 180.0)),
    ("adelante", (0.0, 25.0, 90.0, 180.0)),
    ("derecha", (35.0, 0.0, 90.0, 180.0)),
    ("atrás", (0.0, -20.0, 90.0, 180.0)),
]
ERR_MIN_DEV_DEG = 45.0              # el error queda fuera de este umbral respecto al destino correcto
ERR_MOVE_DEG = (35.0, 60.0)         # recorrido del error (máx. articulación) desde la postura actual
ERR_LIM = ((-80.0, 80.0), (-25.0, 45.0), (50.0, 130.0))  # base, hombro, codo: dentro de los límites de C3
ERR_HAND = 180.0                    # la pinza no se mueve


def random_error_pose(rng, cur, correct):
    """Postura aleatoria (base, hombro, codo) a >= ERR_MIN_DEV_DEG del destino correcto y con un
    recorrido ERR_MOVE_DEG desde la actual; pinza fija."""
    best = None
    for _ in range(5000):
        e = tuple(rng.uniform(lo, hi) for lo, hi in ERR_LIM) + (ERR_HAND,)
        dev = max(abs(x - y) for x, y in zip(e[:3], correct[:3]))
        mv = max(abs(x - y) for x, y in zip(e[:3], cur[:3]))
        if dev >= ERR_MIN_DEV_DEG and ERR_MOVE_DEG[0] <= mv <= ERR_MOVE_DEG[1]:
            return tuple(round(x, 1) for x in e)
        if dev >= ERR_MIN_DEV_DEG and (best is None or abs(mv - ERR_MOVE_DEG[1]) < best[0]):
            best = (abs(mv - ERR_MOVE_DEG[1]), e)
    return tuple(round(x, 1) for x in best[1])


class Arm:
    """Supervisor de C3 (firmware/arm_esp32) por USB: consola de texto a 115200. Reutiliza la
    clase Sup de firmware/arm_esp32/tools/move_ab.py. El pulso de sync lo saca el supervisor
    (GPIO25, alto 1 ms) en el primer paso de cada "act"; ese cable va al GPIO18 del puente."""

    def __init__(self, port, vel, echo=False, seed=0):
        sys.path.insert(0, str(ROOT / "firmware" / "arm_esp32" / "tools"))
        from move_ab import Sup  # noqa: E402

        self.vel = vel
        self.rng = random.Random(seed + 7)  # posturas de error reproducibles con la semilla de la sesión
        self.sup = Sup(port, echo)  # reinicia el supervisor (RTS)
        if not self.sup.wait_for(r"arm_uart: enlace UART", 8, 0):  # desde la apertura del puerto
            sys.exit("el supervisor no arrancó o no tiene el firmware de INTUNE con enlace UART")
        time.sleep(1.0)
        t_end = time.time() + 30  # el brazo tarda unos segundos en arrancar tras encenderlo
        while self.sup.real_pose() is None:  # como move_ab.py: ¿responde el brazo por UART?
            if time.time() > t_end:
                sys.exit("el brazo no responde por UART tras 30 s. ¿Está encendido (con su fuente)? Cables: "
                         "pin 10 del brazo -> RX2/GPIO16, pin 8 -> TX2/GPIO17, GND -> GND del supervisor")
            print("Brazo: no responde todavía; esperando a que arranque (máx. 30 s)...", flush=True)
            time.sleep(2.0)
        m0 = self.sup.mark()  # marcar ANTES de mandar: la respuesta puede llegar enseguida
        self.sup.cmd("s3 0")  # sin S3: simular el heartbeat en nivel 0 (si no, el brazo no se mueve)
        if not self.sup.wait_for(r"-> RUN", 2, m0):
            sys.exit("la seguridad del supervisor no pasó a RUN")
        self.sup.cmd(f"vel {vel}")
        print("Brazo: home...")
        m0 = self.sup.mark()
        self.sup.cmd("home")
        m = self.sup.wait_for(r"HOMING -> IDLE|home rechazado: (.*)", 15, m0)
        if not m or m.group(1):
            sys.exit("home no terminó" + (f": {m.group(1)}" if m else "") + "\n  " + "\n  ".join(self.sup.tail(m0)))
        self.missed = False          # la acción anterior fue un error: la próxima correcta es un regreso
        self.idx = 0                 # postura del anillo donde "debería" estar el brazo
        self.cur = RING[0][1]        # postura real actual
        self._go(RING[0][1])  # postura inicial del anillo, sin pulso
        self.mark = self.sup.mark()

    @staticmethod
    def _rad(pose):
        return " ".join(f"{math.radians(x):.4f}" for x in pose)

    def _go(self, pose):
        m0 = self.sup.mark()
        self.sup.cmd(f"go {self._rad(pose)}")
        self.sup.wait_for(r"motion: MOVING -> IDLE|go rechazado", 15, m0)
        time.sleep(0.5)

    def act(self, err):
        """Lanza la acción. Correcta: siguiente postura del anillo (o la que se saltó en el error
        anterior). Error: postura aleatoria fuera del umbral. Devuelve (nombre, postura en grados,
        regreso): regreso = es la primera correcta tras uno o más errores (vuelve al punto saltado; es
        más larga y visualmente distinta: v3.1 la etiqueta "recovery" y la saca del dataset)."""
        nxt = (self.idx + 1) % len(RING)
        if err:
            pose = random_error_pose(self.rng, self.cur, RING[nxt][1])
            name = "aleatoria"      # self.idx no avanza: la próxima correcta va a RING[nxt]
        else:
            self.idx = nxt
            name, pose = RING[nxt]
        recovery = (not err) and self.missed
        self.missed = bool(err)
        self.cur = pose
        self.mark = self.sup.mark()
        self.sup.cmd(f"act {self._rad(pose)} {1 if err else 0}")
        return name, pose, recovery

    def events(self):
        """Líneas nuevas del supervisor como (t, ONSET | DONE | ERR, texto)."""
        with self.sup.lock:
            new = self.sup.lines[self.mark:]
            self.mark = len(self.sup.lines)
        out = []
        t = time.time()
        for ln in new:
            if "actions: PULSO acción" in ln:
                out.append((t, "ONSET", ln))
            elif "SIN pulso" in ln:
                out.append((t, "ERR", ln))
            elif "MOVING -> IDLE" in ln:
                out.append((t, "DONE", ln))
            elif "rechazado" in ln:
                out.append((t, "ERR", ln))
        return out

    def close(self):
        with contextlib.suppress(Exception):
            self.sup.cmd("stop")
            time.sleep(0.3)
            self.sup.cmd("s3 off")  # sin heartbeat el supervisor retiene el brazo (quieto, con torque)
            time.sleep(0.5)
            # el puerto no se cierra aquí: el hilo lector de Sup (daemon) fallaría al leer de un
            # puerto cerrado; se cierra solo al salir el proceso


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


def pause(msg):
    """input() que no falla sin teclado (stdin cerrado): entonces sigue sin esperar."""
    try:
        return input(msg)
    except EOFError:
        return ""


def new_state(total):
    return {"next_id": 1, "total": total, "last_seq": None,
            "stats": {"ok": 0, "no_pulse": 0, "no_t0": 0, "overlap": 0, "arm_err": 0}}


def run_actions(a, br, arm, items, we, rng, st, rest_every=0):
    """items: lista de (etiqueta, err). Etiquetas: correct / error (dataset), familiar / check (sin
    errores, fuera del dataset: build_dataset las deja con y = -1). Devuelve las estadísticas de ESTE
    segmento; st acumula id de acción, seq de flancos y totales entre segmentos."""
    n = st["total"]
    stats = st["stats"]
    seg = {k: 0 for k in stats}
    last_seq = st["last_seq"]
    for j, (seg_label, err) in enumerate(items):
        i = st["next_id"]
        st["next_id"] += 1
        if rest_every and j > 0 and j % rest_every == 0:
            pause(f"\nDescanso ({i - 1}/{n}). Enter para seguir... ")
        # vaciar eventos sueltos (rebotes, pulsos fuera de acción)
        while not br.events.empty():
            br.events.get_nowait()
        t_sent = time.time()
        rest_before = t_sent - st["t_still"] if st.get("t_still") else None
        label = "manual" if arm is None else seg_label
        target, pose, recovery = "", None, False
        if arm is None:
            print(f"\nACCIÓN {i}/{n}: da un pulso en GPIO18 ahora", end="", flush=True)
        else:
            target, pose, recovery = arm.act(err)
            if recovery and label == "correct":
                label = "recovery"  # regreso al punto saltado: fuera del dataset (y = -1)
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
                for t_ln, kind, ln in arm.events():
                    if kind == "ONSET" and t_onset is None:
                        t_onset = t_ln
                    elif kind == "DONE":
                        t_done = t_ln
                    elif kind == "ERR":
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
        seg[status] += 1
        seq, cnt, off, fl = evt[1] if evt else ("", "", "", "")
        if evt and last_seq is not None and seq != last_seq + 1:
            print(f"\n!! saltaron flancos: seq {last_seq} -> {seq}", file=sys.stderr)
        if evt:
            last_seq = seq
        PUB.send({"k": "act", "i": i, "n": n, "label": label, "status": status, "target": target,
                  "counter": cnt if cnt != "" else None})
        we.writerow([i, label, f"{t_sent:.4f}", f"{t_onset:.4f}" if t_onset else "",
                     f"{t_done:.4f}" if t_done else "", seq, cnt, off, fl, status, arm_msg,
                     target, " ".join(f"{x:g}" for x in pose) if pose else "",
                     f"{rest_before:.3f}" if rest_before is not None else ""])
        print(f"\r[{i:3d}/{n}] {label:8s} {target:9s} {status:8s} cnt={cnt}   "
              + " ".join(f"{k}={v}" for k, v in stats.items()), end="", flush=True)
        t_ref = evt[0] if evt else t_sent
        if arm is not None and t_done:
            # v3.1: reposo medido desde que el brazo se DETIENE, igual para todas las acciones. Así la
            # respuesta cerebral al frenado (~300-600 ms) nunca cae en la línea base [-200, 0] ms de la
            # siguiente época, sea cual sea la duración del movimiento anterior.
            wait = t_done + rng.uniform(*a.rest) - time.time()
        else:
            wait = rng.uniform(*a.isi) - (time.time() - t_ref)  # sin brazo: intervalo entre inicios
        wait = max(wait, t_ref + 1.5 - time.time())  # CLAUDE.md: >= 1.5 s entre inicios
        st["t_still"] = t_done
        if wait > 0:
            time.sleep(wait)
    print()
    st["last_seq"] = last_seq
    return seg


CHECK_FIELDS = ["block", "try", "passed", "t_start", "t_end", "pulses_ok", "n_actions", "lost", "loss_pct"] + \
    [f"rstd_{c}" for c in L.CH]


def check_quality(a, br, t0, t1, seg, n_act, lost0, block, attempt, wc):
    """Calidad de las acciones de prueba: desviación robusta (1.4826 * MAD) del EEG filtrado por canal
    (un parpadeo suelto no la mueve, un electrodo flojo sí), pulsos válidos y muestras perdidas."""
    import numpy as np

    rows = [(fl, f) for t, fl, f in list(br.recent) if t0 <= t <= t1 and not (fl & 0x07)]
    if len(rows) < L.FS * 10:
        print(f"  prueba: solo {len(rows)} muestras limpias; no se puede evaluar")
        rstd = np.full(8, np.inf)
    else:
        x = np.array([f for _, f in rows])
        rstd = 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    lost = (br.last_status[5] - lost0) if br.last_status else 0
    total = len(rows) + lost
    loss_pct = 100.0 * lost / total if total else 100.0
    key = [L.CH.index(c) for c in a.check_channels]
    bad = [L.CH[i] for i in key if rstd[i] > a.check_max_uv]
    warn_ch = [L.CH[i] for i in range(8) if i not in key and rstd[i] > a.check_max_uv]
    passed = not bad and seg["ok"] >= n_act - 1 and loss_pct < 1.0
    print("  ruido robusto (µV): " + "  ".join(f"{c}={v:.1f}{'!' if v > a.check_max_uv else ''}"
                                               for c, v in zip(L.CH, rstd)))
    print(f"  pulsos ok {seg['ok']}/{n_act}, pérdida {loss_pct:.2f} %  ->  {'BIEN' if passed else 'MAL'}"
          + (f"  (canales clave malos: {', '.join(bad)})" if bad else "")
          + (f"  (aviso, otros canales: {', '.join(warn_ch)})" if warn_ch else ""))
    wc.writerow([block, attempt, int(passed), f"{t0:.3f}", f"{t1:.3f}", seg["ok"], n_act, lost, f"{loss_pct:.3f}"]
                + [f"{v:.2f}" for v in rstd])
    PUB.send({"k": "check", "block": block, "try": attempt, "passed": bool(passed),
              "rstd": [round(float(v), 2) if np.isfinite(v) else None for v in rstd], "max_uv": a.check_max_uv})
    return passed


def run_check(a, br, arm, we, rng, st, block, wc):
    """Prueba de calidad antes de un bloque del dataset; se repite si sale mal."""
    for attempt in range(1, a.check_max_tries + 1):
        print(f"\n--- prueba de señal antes del bloque {block} (intento {attempt}): {a.check_n} acciones sin errores")
        PUB.send({"k": "phase", "text": f"prueba de señal antes del bloque {block} (intento {attempt})"})
        lost0 = br.last_status[5] if br.last_status else 0
        t0 = time.time()
        seg = run_actions(a, br, arm, [("check", 0)] * a.check_n, we, rng, st)
        time.sleep(1.0)  # que entren las muestras de la última época
        if check_quality(a, br, t0, time.time(), seg, a.check_n, lost0, block, attempt, wc):
            return True
        if attempt < a.check_max_tries:
            try:
                ans = input("  Señal insuficiente: acomoda la diadema / pon gel en los canales marcados.\n"
                            "  Enter = repetir la prueba, c = seguir igual, q = terminar: ").strip().lower()
            except EOFError:  # sin teclado (stdin cerrado): seguir
                ans = "c"
            if ans == "c":
                return False
            if ans == "q":
                raise KeyboardInterrupt
    print(f"  {a.check_max_tries} intentos sin pasar: se sigue igual (queda anotado en checks.csv)")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--bridge-port", help="por defecto el primer /dev/ttyUSB*")
    ap.add_argument("--arm-port", help="supervisor del brazo (C3), p. ej. /dev/serial/by-id/...")
    ap.add_argument("--vel", type=float, default=17.5,
                    help="°/s del brazo (def. 17.5: el tramo de 35° dura ~2 s, CLAUDE.md pide acciones >= ~2 s)")
    ap.add_argument("--arm-echo", action="store_true", help="mostrar el log del supervisor")
    ap.add_argument("--no-arm", action="store_true", help="banco: pulsos manuales, sin brazo")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--n-actions", type=int, default=300)
    ap.add_argument("--p-error", type=float, default=0.22)
    ap.add_argument("--isi", type=float, nargs=2, default=(3.0, 4.0), metavar=("MIN", "MAX"),
                    help="s entre inicios de acción (>= 1.5 s y >= duración de la acción)")
    ap.add_argument("--rest-every", type=int, default=50, help="descanso cada N acciones (sin --check-every)")
    ap.add_argument("--familiar", type=int, default=0, metavar="N",
                    help="N acciones sin errores al inicio para memorizar el recorrido (label familiar)")
    ap.add_argument("--familiar-pause", type=float, default=30.0, help="s de pausa tras la familiarización")
    ap.add_argument("--check-every", type=int, default=0, metavar="N",
                    help="prueba de señal antes de cada bloque de N acciones del dataset (0 = no)")
    ap.add_argument("--check-n", type=int, default=20, help="acciones de cada prueba (label check)")
    ap.add_argument("--check-max-uv", type=float, default=20.0, help="ruido robusto máximo por canal clave (µV)")
    ap.add_argument("--check-channels", default="Fz,Cz,C3,C4,Pz", type=lambda v: v.split(","),
                    help="canales que deben pasar la prueba")
    ap.add_argument("--check-max-tries", type=int, default=3)
    ap.add_argument("--ui-port", type=int, default=47474,
                    help="puerto UDP local para tools/recording/live_view.py (0 = no publicar)")
    ap.add_argument("--rest", type=float, nargs=2, default=(1.5, 2.5), metavar=("MIN", "MAX"),
                    help="s quieto desde que TERMINA un movimiento hasta que empieza el siguiente (def. 1.5-2.5)")
    ap.add_argument("--no-block-rest", dest="block_rest", action="store_false",
                    help="sin descanso (Enter) al terminar cada bloque de --check-every")
    ap.add_argument("--pulse-timeout", type=float, default=3.0, help="s para recibir el flanco")
    ap.add_argument("--action-timeout", type=float, default=10.0,
                    help="s tras el flanco para DONE (el regreso tras un error puede durar ~5 s)")
    ap.add_argument("--min-rest", type=float, default=0.8, help="s mínimos quieto entre acciones")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default=str(ROOT / "ml" / "data" / "raw"))
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    bad_ch = [c for c in a.check_channels if c not in L.CH]
    if bad_ch:
        sys.exit(f"--check-channels: canales desconocidos {bad_ch}; válidos {L.CH}")
    if a.isi[0] < 1.5:
        sys.exit("--isi mínimo 1.5 s (CLAUDE.md)")
    if not a.no_arm and not a.arm_port:
        sys.exit("falta --arm-port (o --no-arm para la prueba de banco)")
    if not a.no_arm and not a.bridge_port:
        sys.exit("con el brazo hay dos placas por USB: indica también --bridge-port")

    seed = a.seed if a.seed is not None else int(time.time())
    rng = random.Random(seed)
    labels = make_schedule(a.n_actions, a.p_error, rng)
    outdir = Path(a.out) / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.subject}"
    outdir.mkdir(parents=True)

    from design_iir import BAND, FS, N
    meta = {
        "subject": a.subject, "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
        "n_actions": a.n_actions, "p_error": a.p_error, "isi_s": list(a.isi), "no_arm": a.no_arm,
        "familiar": a.familiar, "check_every": a.check_every, "check_n": a.check_n,
        "check_max_uv": a.check_max_uv, "check_channels": a.check_channels,
        "paradigm": None if a.no_arm else {
            "version": "3.1",
            "recovery": "la primera correcta tras un error (regreso al punto saltado) se etiqueta recovery "
                        "y queda fuera del dataset",
            "rest_after_done_s": list(a.rest),
            "rule": "correcta = siguiente postura del anillo (o la saltada por el error anterior); "
                    "error = postura aleatoria a >= err_min_dev_deg del destino correcto",
            "ring_deg": {name: list(pose) for name, pose in RING}, "vel_deg_s": a.vel,
            "err_min_dev_deg": ERR_MIN_DEV_DEG, "err_move_deg": list(ERR_MOVE_DEG),
            "err_lim_deg": [list(x) for x in ERR_LIM]},
        "filter": f"scipy.signal.butter({N}, {list(BAND)}, btype='band', fs={FS:g}, output='sos'), causal",
        "channels": L.CH, "fs": L.FS, "sync": "flanco GPIO del brazo -> puente GPIO18 -> trama EVENT",
        "notes": a.notes,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Sesión: {outdir}")

    global PUB
    PUB = Publisher(a.ui_port)
    br = Bridge(a.bridge_port, outdir)
    br.start()
    arm = None if a.no_arm else Arm(a.arm_port, a.vel, a.arm_echo, seed)
    stats = {}
    try:
        wait_ready(br)
        if not a.no_blocks:
            with open(outdir / "blocks.csv", "w", newline="") as fb:
                wb = csv.writer(fb)
                wb.writerow(["block", "counter_start", "counter_end"])
                run_blocks(br, wb)
        if arm is not None:
            print("\nINSTRUCCIÓN AL OPERADOR: el brazo recorre siempre " + " -> ".join(n for n, _ in RING)
                  + " -> ... A veces se EQUIVOCA y se va a cualquier otro lado; después vuelve al punto que "
                  "le tocaba. Solo obsérvalo, sin moverte.")
        input(f"\nParadigma: {a.n_actions} acciones, ~{a.p_error:.0%} errores. Observa el brazo. "
              "Enter para empezar... ")
        with open(outdir / "events.csv", "w", newline="") as fe:
            we = csv.writer(fe)
            we.writerow(["action_id", "label", "t_sent", "t_onset_arm", "t_done_arm",
                         "evt_seq", "evt_counter", "evt_offset_us", "evt_flags", "status", "arm_msg",
                         "target", "pose_deg", "rest_before_s"])
            n_extra = a.familiar + (a.check_n * -(-a.n_actions // a.check_every) if a.check_every else 0)
            st = new_state(a.n_actions + n_extra)
            stats = st["stats"]
            if a.familiar:
                print(f"\n--- familiarización: {a.familiar} acciones sin errores (memoriza el recorrido)")
                PUB.send({"k": "phase", "text": "familiarización"})
                run_actions(a, br, arm, [("familiar", 0)] * a.familiar, we, rng, st)
                print(f"--- pausa de {a.familiar_pause:.0f} s")
                PUB.send({"k": "phase", "text": f"pausa de {a.familiar_pause:.0f} s"})
                time.sleep(a.familiar_pause)
            items = [("error" if e else "correct", e) for e in labels]
            if a.check_every:
                with open(outdir / "checks.csv", "w", newline="") as fc:
                    wc = csv.writer(fc)
                    wc.writerow(CHECK_FIELDS)
                    for b0 in range(0, a.n_actions, a.check_every):
                        block = b0 // a.check_every + 1
                        run_check(a, br, arm, we, rng, st, block, wc)
                        fc.flush()
                        print(f"\n--- bloque {block}: acciones {b0 + 1}-{min(b0 + a.check_every, a.n_actions)} del dataset")
                        PUB.send({"k": "phase", "text": f"bloque {block} del dataset"})
                        run_actions(a, br, arm, items[b0:b0 + a.check_every], we, rng, st)
                        fe.flush()
                        if a.block_rest and b0 + a.check_every < a.n_actions:
                            PUB.send({"k": "phase", "text": f"DESCANSO tras el bloque {block} (Enter en la terminal)"})
                            pause(f"\n=== DESCANSO tras el bloque {block} ({b0 + a.check_every}/{a.n_actions} del "
                                  "dataset). Relájate, parpadea, muévete si quieres. Enter para seguir... ")
                            st["t_still"] = None  # tras la pausa, el reposo previo no es comparable
            else:
                run_actions(a, br, arm, items, we, rng, st, rest_every=a.rest_every)
    except KeyboardInterrupt:
        print("\nInterrumpido; se guarda lo grabado.")
    finally:
        if arm is not None:
            arm.close()
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
