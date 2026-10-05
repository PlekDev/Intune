#!/usr/bin/env python3
"""INTUNE C3: mueve el brazo de A a B (y de vuelta) a través del supervisor.

Cada movimiento es una ACCIÓN: al salir su primer paso, el supervisor sube el pulso de sync
(GPIO25, activo en alto ~1 ms, bajo en reposo) hacia el S3 y le manda un EVENT. Con --sin-pulso
se mueve igual pero sin pulso. Entre pulsos siempre hay al menos 1.5 s (paradigma de C4).

Los ángulos van en GRADOS, en el orden base, hombro, codo, pinza.
La postura de arranque del brazo es 0,0,90,180 (hombro vertical, antebrazo horizontal).

  move_ab.py                                      # A y B por defecto, 3 idas y vueltas
  move_ab.py --a -30,0,90,180 --b 30,15,100,180 --reps 5 --vel 25 --pausa 1.5

Qué hace:
  1. Reinicia el supervisor, comprueba que el brazo responde (posición real por UART).
  2. Simula el S3 en nivel 0 (sin S3 el supervisor no mueve nada) y hace home.
  3. Repite: va a A, espera, va a B, espera. Muestra la posición real en cada punto.
  4. Vuelve a la postura de arranque y retiene el brazo (simulación del S3 apagada).
Ctrl+C: stop inmediato y brazo retenido.

Límites (los aplica también el supervisor): base ±91°, hombro -34..51°, codo 34..149°, pinza 92..183°.
"""
import argparse
import math
import re
import sys
import threading
import time

import serial
from sup_console import find_port

INIT = (0.0, 0.0, 90.0, 180.0)
LIM = ((-91, 91), (-34, 51), (34, 149), (92, 183))
NAMES = ('base', 'hombro', 'codo', 'pinza')


class Sup:
    def __init__(self, port, echo):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.05
        self.s.dtr = self.s.rts = False
        self.s.open()
        self.lines = []
        self.lock = threading.Lock()
        self.echo = echo
        threading.Thread(target=self._read, daemon=True).start()
        self.s.rts = True; time.sleep(0.1); self.s.rts = False   # reinicio limpio

    def _read(self):
        buf = b''
        while True:
            buf += self.s.read(512)
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode(errors='replace').strip()
                with self.lock:
                    self.lines.append(line)
                if self.echo and re.search(r' (sup|motion|safety|arm_uart): ', line):
                    print('   ' + line, flush=True)

    def cmd(self, c):
        self.s.write(c.encode() + b'\n')

    def mark(self):
        with self.lock:
            return len(self.lines)

    def wait_for(self, pattern, timeout, start=None):
        """Espera una línea nueva (desde start, o desde ahora) que cumpla pattern. Devuelve el match o None."""
        if start is None:
            start = self.mark()
        t_end = time.time() + timeout
        rx = re.compile(pattern)
        while time.time() < t_end:
            with self.lock:
                new = self.lines[start:]
            for i, l in enumerate(new):
                m = rx.search(l)
                if m:
                    return m
            time.sleep(0.02)
        return None

    def real_pose(self):
        self.cmd('status')
        m = self.wait_for(r'posición REAL del brazo \(hace (\d+) ms\): b=([-+\d.]+) s=([-+\d.]+) e=([-+\d.]+) t=([-+\d.]+)', 2)
        if not m:
            return None
        return tuple(math.degrees(float(m.group(i))) for i in range(2, 6))


def parse_pose(text):
    v = tuple(float(x) for x in text.split(','))
    if len(v) != 4:
        raise argparse.ArgumentTypeError('hacen falta 4 ángulos: base,hombro,codo,pinza')
    for x, (lo, hi), n in zip(v, LIM, NAMES):
        if not lo <= x <= hi:
            raise argparse.ArgumentTypeError(f'{n}={x}° fuera de {lo}..{hi}°')
    return v


def fmt(p):
    return ' '.join(f'{n}={x:+6.1f}°' for n, x in zip(NAMES, p))


_last = list(INIT)
_last_pulse = [0.0]
MIN_SPACING_S = 1.5


def go(sup, pose, label, vel, pulse=False):
    rad = ' '.join(f'{math.radians(x):.4f}' for x in pose)
    dur = max(abs(p - q) for p, q in zip(pose, _last)) / vel   # la articulación que más recorre
    if pulse:
        wait = _last_pulse[0] + MIN_SPACING_S - time.time()
        if wait > 0:
            time.sleep(wait)
    start = sup.mark()
    sup.cmd(f'{"act" if pulse else "go"} {rad}')
    m = sup.wait_for(r'(go|act) rechazado: (.*)', 0.3, start)
    if m:
        raise SystemExit(f'el supervisor rechazó {label}: {m.group(2)}')
    if pulse:
        if sup.wait_for(r'actions: PULSO acción', 0.5, start):
            _last_pulse[0] = time.time()
            print(f'{label}: PULSO de sync enviado', flush=True)
        else:
            print(f'{label}: AVISO, no se vio el pulso de sync', flush=True)
    if dur > 0.05 and not sup.wait_for(r'motion: MOVING -> IDLE', dur + 3, start):
        print(f'{label}: AVISO, el movimiento no terminó en {dur + 3:.1f} s', flush=True)
    _last[:] = pose
    time.sleep(0.4)   # que el servo asiente y llegue una lectura nueva
    real = sup.real_pose()
    err = '' if real is None else '  error ' + ' '.join(f'{r - p:+.1f}' for r, p in zip(real, pose))
    print(f'{label}: pedido {fmt(pose)}')
    print(f'{" " * len(label)}  real   {fmt(real) if real else "(sin lectura)"}{err}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', type=parse_pose, default=(-30.0, 0.0, 90.0, 180.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--b', type=parse_pose, default=(30.0, 10.0, 100.0, 180.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--reps', type=int, default=3, help='idas y vueltas A->B->A')
    ap.add_argument('--vel', type=float, default=20.0, help='velocidad máxima por articulación, °/s (máx 57)')
    ap.add_argument('--pausa', type=float, default=1.0, help='segundos quieto en A y en B')
    ap.add_argument('--sin-pulso', action='store_true', help='mover sin pulso de sync (solo go)')
    ap.add_argument('-v', '--verbose', action='store_true', help='mostrar el log del supervisor')
    ap.add_argument('-p', '--port')
    a = ap.parse_args()

    sup = Sup(a.port or find_port(), a.verbose)
    try:
        print('Reiniciando supervisor...', flush=True)
        if not sup.wait_for(r'arm_uart: enlace UART', 8):
            raise SystemExit('el supervisor no arrancó (¿USB?)')
        time.sleep(1.0)
        real = sup.real_pose()
        if real is None:
            raise SystemExit('el brazo no responde por UART: ¿encendido? ¿cables pin 10 -> D16/RX2, pin 8 -> D17/TX2, GND?')
        print(f'Brazo responde. Posición real: {fmt(real)}')

        sup.cmd('s3 0')
        if not sup.wait_for(r'-> RUN', 2):
            raise SystemExit('la seguridad no pasó a RUN')
        sup.cmd(f'vel {min(a.vel, 57)}')
        a.vel = min(a.vel, 57)
        print('Home (postura de arranque, lento)...', flush=True)
        start = sup.mark()
        sup.cmd('home')
        m = sup.wait_for(r'HOMING -> IDLE|home rechazado: (.*)', 10, start)
        if not m or m.group(1):
            with sup.lock:
                tail = [l for l in sup.lines[start:] if re.search(r' (sup|motion|safety|actions): ', l)][-8:]
            raise SystemExit('home no terminó' + (f': {m.group(1)}' if m else '') + '\n  ' + '\n  '.join(tail))

        print(f'A = {fmt(a.a)}\nB = {fmt(a.b)}\n{a.reps} idas y vueltas a {a.vel:.0f}°/s, pausa {a.pausa} s\n', flush=True)
        for i in range(a.reps):
            go(sup, a.a, f'[{i + 1}] A', a.vel, not a.sin_pulso)
            time.sleep(a.pausa)
            go(sup, a.b, f'[{i + 1}] B', a.vel, not a.sin_pulso)
            time.sleep(a.pausa)

        print('\nVolviendo a la postura de arranque...', flush=True)
        go(sup, INIT, 'inicio', a.vel)
    except KeyboardInterrupt:
        print('\nCtrl+C: stop', flush=True)
        sup.cmd('stop')
        time.sleep(0.3)
    finally:
        sup.cmd('s3 off')   # sin heartbeat el supervisor retiene el brazo
        time.sleep(0.5)
        print('Brazo retenido (simulación del S3 apagada).')


if __name__ == '__main__':
    main()
