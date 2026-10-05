#!/usr/bin/env python3
"""INTUNE C3: diagnóstico de ESP-NOW supervisor -> RoArm-M2-S (sin intervención).

Requiere: PC conectado al Wi-Fi "RoArm-M2" (HTTP para leer config y posición) y el
supervisor por USB (manda los comandos ESP-NOW). Todo queda en logs/espnow_<fecha>.log.

  1. Lee por HTTP la config del brazo (T:405 Wi-Fi, T:302 MAC, T:602 misión de arranque).
  2. Prueba 4 variantes ESP-NOW (unicast/broadcast x cmd 1/2): gira la base 0.2 rad
     (~11°, 18°/s) y comprueba con T:105 si se movió; si se movió, vuelve.
  3. Si ninguna funcionó: activa en RAM modo follower y broadcast (T:301 mode 3,
     T:300 mode 1; se pierde al reiniciar el brazo) y repite.

Ctrl+C: ordena por HTTP mantener la posición actual.
"""
import datetime
import glob
import json
import os
import re
import sys
import threading
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roarm_http as rh  # noqa: E402

STEP = 0.2
_log = None


def out(msg=''):
    line = f'{datetime.datetime.now():%H:%M:%S.%f}'[:-3] + '  ' + msg
    print(line, flush=True)
    if _log:
        _log.write(line + '\n')
        _log.flush()


def redact(body):
    return re.sub(r'("[a-z_]*password"\s*:\s*)"[^"]*"', r'\1"***"', body)


def http(cmd):
    body, ms = rh.send(cmd)
    out(f'HTTP {json.dumps(cmd)} ({ms:.0f} ms) -> {redact(body)}')
    return body


def base_now():
    return json.loads(rh.send({'T': 105})[0])['b']


class Supervisor:
    def __init__(self):
        ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
        if not ports:
            raise SystemExit('supervisor no conectado por USB')
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = ports[0], 115200, 0.1
        self.s.dtr = self.s.rts = False
        self.s.open()
        self.lines = []
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        self.s.rts = True; time.sleep(0.1); self.s.rts = False

    def _read(self):
        buf = b''
        while True:
            buf += self.s.read(256)
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                text = line.decode(errors='replace').strip()
                if ' sup: ' in text:
                    out(f'  SUP {text}')
                    with self.lock:
                        self.lines.append(text)

    def cmd(self, c, wait=0.3):
        self.s.write(c.encode() + b'\n')
        time.sleep(wait)

    def wait_found(self, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if any('scan: "' in l and 'canal' in l for l in self.lines):
                    return True
            time.sleep(0.2)
        return False


def try_variants(sup, label):
    out(f'=== {label} ===')
    worked = []
    for peer, cmd in (('ap', 2), ('bcast', 2), ('bcast', 1), ('ap', 1)):
        b0 = base_now()
        tgt = b0 + (STEP if b0 <= 0 else -STEP)
        sup.cmd(f'peer {peer}')
        sup.cmd(f'cmd {cmd}')
        sup.cmd(f'base {tgt:.3f}', wait=2.5)
        b1 = base_now()
        moved = abs(b1 - b0) > 0.1
        out(f'RESULTADO {label} peer={peer} cmd={cmd}: b {b0:+.3f} -> {b1:+.3f}  {"SE MOVIÓ" if moved else "no se movió"}')
        if moved:
            worked.append((peer, cmd))
            sup.cmd(f'base {b0:.3f}', wait=2.5)
    return worked


def main():
    global _log
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, f'espnow_{datetime.datetime.now():%Y%m%d_%H%M%S}.log')
    _log = open(path, 'w')
    print(f'log: {path}')
    try:
        try:
            http({'T': 105})
        except OSError as e:
            out(f'SIN CONEXIÓN HTTP con {rh.HOST}: {e}. ¿PC conectado a "RoArm-M2"?')
            return
        out('=== Config del brazo ===')
        for c in ({'T': 405}, {'T': 302}, {'T': 602}):
            try:
                http(c)
            except OSError as e:
                out(f'HTTP {c} ERROR {e}')

        sup = Supervisor()
        if not sup.wait_found():
            out('el supervisor no encontró el AP del brazo: acércalo y repite')
            return
        time.sleep(1)

        ok = try_variants(sup, 'Ronda 1 (config actual)')
        if not ok:
            out('=== Activando en RAM: follower + broadcast ===')
            http({'T': 301, 'mode': 3})
            http({'T': 300, 'mode': 1, 'mac': 'FF:FF:FF:FF:FF:FF'})
            ok = try_variants(sup, 'Ronda 2 (follower + broadcast)')
        sup.cmd('status', wait=0.5)
        out(f'=== FIN: variantes que funcionaron: {ok or "ninguna"} ===')
    except KeyboardInterrupt:
        out('Ctrl+C: HOLD por HTTP')
        try:
            out(f'HOLD en b={rh.hold_here():+.3f}')
        except Exception as e:
            out(f'HOLD falló: {e}. Usa el interruptor si hace falta.')
    except Exception as e:
        out(f'ERROR: {type(e).__name__}: {e}')
    finally:
        print(f'\nlog guardado en: {path}')


if __name__ == '__main__':
    main()
