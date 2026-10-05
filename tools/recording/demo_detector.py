#!/usr/bin/env python3
"""
INTUNE: demo en vivo del detector ErrP local (supervisor con CONFIG_ARM_LOCAL_DETECTOR).

El brazo recorre el anillo izquierda -> adelante -> derecha -> atrás; con probabilidad --p-error
hace un error deliberado (postura aleatoria a >= 45° del destino, como en la grabación v3.1) y la
siguiente acción vuelve al punto saltado. El supervisor recibe el EEG del puente por UART, corta la
época de cada acción y decide el nivel con el ErrP-AE; este script solo ordena las acciones y
muestra lo que decide el supervisor: score, nivel y reacción del brazo.

    .venv/bin/python tools/recording/demo_detector.py --port /dev/serial/by-id/<supervisor>

Cableado: puente GPIO17 -> supervisor GPIO26 (EEG), supervisor GPIO25 -> puente GPIO18 (pulso),
GND común; brazo en pin 10 -> GPIO16, pin 8 -> GPIO17. Diadema puesta y encendida, puente cerca.
Teclas: Ctrl+C termina (el brazo queda quieto con torque).
Si el detector pausa (nivel 2) o detiene (nivel 3) el brazo, el script espera --resume-after s (def.
5) y luego confirma / rearma solo, y el brazo sigue. Con --manual espera el botón BOOT (corto =
confirmar la pausa, largo = rearmar tras parada). Sin EEG el rearme sigue bloqueado (fail-safe).
"""

import argparse
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "C3" / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "recording"))
from move_ab import Sup  # noqa: E402
import record_errp as R  # noqa: E402

LEVEL_TXT = {0: "0 normal", 1: "1 LEVE (lento)", 2: "2 MODERADO (pausa)", 3: "3 GRAVE (parada)"}


def rad(p):
    return " ".join(f"{math.radians(x):.4f}" for x in p)


def safety_state(s):
    m0 = s.mark()
    s.cmd("status")
    m = s.wait_for(r"seguridad (\w+) \|", 1.5, m0)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="supervisor (p. ej. /dev/serial/by-id/...)")
    ap.add_argument("--p-error", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=0, help="acciones (0 = hasta Ctrl+C)")
    ap.add_argument("--vel", type=float, default=17.5)
    ap.add_argument("--rest", type=float, nargs=2, default=(1.5, 2.5), help="s quieto entre acciones")
    ap.add_argument("--resume-after", type=float, default=5.0,
                    help="s que el brazo queda pausado/parado tras una alerta antes de seguir solo (def. 5)")
    ap.add_argument("--manual", action="store_true", help="no seguir solo: esperar el botón BOOT")
    ap.add_argument("--auto-rearm", action="store_true", help="(obsoleto) igual que --resume-after 0")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    s = Sup(a.port, False)
    if not s.wait_for(r"arm_uart: enlace UART", 8, 0):
        sys.exit("el supervisor no arrancó")
    if not s.wait_for(r"det: ErrP-AE", 3, 0):
        sys.exit("el supervisor no tiene el detector local (CONFIG_ARM_LOCAL_DETECTOR)")
    print("Esperando el EEG del puente (puente cerca de la diadema; si tarda, apaga y prende la diadema)...")
    t0 = time.time()
    while not s.wait_for(r"EEG del puente recibido", 5, 0):
        print(f"  ... {time.time() - t0:.0f} s sin EEG", flush=True)
    print("EEG recibido. Esperando la respuesta del brazo...")
    while s.real_pose() is None:
        print("  ... el brazo no responde por UART (¿encendido? ¿cables?)", flush=True)
        time.sleep(2)
    time.sleep(1.5)
    if safety_state(s) == "STOPPED":
        s.cmd("reset")
        time.sleep(0.5)
    s.cmd(f"vel {a.vel}")
    m0 = s.mark()
    s.cmd("home")
    s.wait_for(r"HOMING -> IDLE", 15, m0)
    m0 = s.mark()
    s.cmd("go " + rad(R.RING[0][1]))
    s.wait_for(r"MOVING -> IDLE", 15, m0)
    time.sleep(2)

    print("\nDEMO: el brazo recorre " + " -> ".join(n for n, _ in R.RING) + ". A veces se equivoca.\n")
    idx, cur, missed, i = 0, R.RING[0][1], False, 0
    stats = {"correct": [0, 0, 0, 0], "error": [0, 0, 0, 0], "rechazada": 0}
    try:
        while a.n <= 0 or i < a.n:
            st = safety_state(s)
            if st not in ("RUN", "SLOW", "PAUSED", "STOPPED"):
                # NO_S3 (sin heartbeat del detector: EEG aún no llega o se perdió) u otro: esperar
                print(f"  >> seguridad {st}: brazo retenido, esperando el EEG / el detector...", flush=True)
                time.sleep(2)
                continue
            if st in ("PAUSED", "STOPPED"):
                if not a.manual:
                    wait = 0.0 if a.auto_rearm else a.resume_after
                    what = "PARADO (nivel 3)" if st == "STOPPED" else "PAUSADO (nivel 2)"
                    for left in range(int(math.ceil(wait)), 0, -1):
                        print(f"  >> brazo {what}: sigue en {left} s   ", end="\r", flush=True)
                        time.sleep(min(1.0, wait))
                    print(f"  >> brazo {what}: {wait:.0f} s cumplidos, {'rearmo' if st == 'STOPPED' else 'confirmo'} y sigue",
                          flush=True)
                    s.cmd("reset" if st == "STOPPED" else "confirm")
                    time.sleep(0.5)
                    if safety_state(s) in ("PAUSED", "STOPPED"):
                        print("  >> no se pudo reanudar (¿sin EEG?); reintento", flush=True)
                        time.sleep(2)
                    continue
                else:
                    print(f"  >> brazo {st}: {'pulsación LARGA de BOOT para rearmar' if st == 'STOPPED' else 'pulsación corta de BOOT para confirmar'}",
                          flush=True)
                time.sleep(2)
                continue
            i += 1
            saved = (idx, cur, missed)  # por si el supervisor rechaza la acción y el brazo no se mueve
            err = rng.random() < a.p_error and not missed
            nxt = (idx + 1) % len(R.RING)
            if err:
                pose, name = R.random_error_pose(rng, cur, R.RING[nxt][1]), "ERROR (aleatoria)"
            else:
                idx = nxt
                name, pose = R.RING[nxt]
            kind = "error" if err else ("regreso" if missed else "correct")
            missed = err
            cur = pose
            m0 = s.mark()
            s.cmd(f"act {rad(pose)} {1 if err else 0}")
            rj = s.wait_for(r"act rechazado: (.*)", 0.4, m0)
            if rj:
                print(f"[{i:3d}] {name}: rechazada por el supervisor ({rj.group(1)}); estado {safety_state(s)}")
                i -= 1  # no cuenta: se repite cuando la seguridad lo permita
                idx, cur, missed = saved
                time.sleep(2)
                continue
            s.wait_for(r"MOVING -> IDLE", 12, m0)
            # la época se decide ~1 s después del inicio (ventana de 800 ms + margen)
            m = s.wait_for(r"det: época cnt=\d+ (score=([\d.]+).*-> nivel (\d)|RECHAZADA \((\w+)\).*nivel (\d))", 3, m0)
            if not m:
                print(f"[{i:3d}] {kind:8s} {name:18s} sin decisión del detector")
            elif m.group(2):
                lvl = int(m.group(3))
                if kind in stats:
                    stats[kind][lvl] += 1
                print(f"[{i:3d}] {kind:8s} {name:18s} score {float(m.group(2)):.3f} -> nivel {LEVEL_TXT[lvl]}", flush=True)
            else:
                stats["rechazada"] += 1
                print(f"[{i:3d}] {kind:8s} {name:18s} época rechazada ({m.group(4)}) -> nivel {m.group(5)}", flush=True)
            time.sleep(rng.uniform(*a.rest))
    except KeyboardInterrupt:
        pass
    finally:
        s.cmd("stop")
        time.sleep(0.3)
        s.cmd("det")
        time.sleep(0.5)
        print("\nNiveles por tipo de acción (0/1/2/3):")
        print(f"  correctas: {stats['correct']}   errores: {stats['error']}   épocas rechazadas: {stats['rechazada']}")
        print("Brazo quieto con torque.")


if __name__ == "__main__":
    main()
