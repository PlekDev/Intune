#!/usr/bin/env python3
"""INTUNE C3: mueve el brazo de A a B y de vuelta, sin fin, a través del supervisor (ESP32 por USB).

Uso (Linux, macOS o Windows; solo necesita pyserial):
  python firmware/arm_esp32/tools/move_ab.py                  # A y B por defecto, hasta Ctrl+C
  python firmware/arm_esp32/tools/move_ab.py --a -45,0,90,180 --b 45,20,110,180 --vel 25 --pausa 2
  python firmware/arm_esp32/tools/move_ab.py --reps 3         # solo 3 idas y vueltas
  python firmware/arm_esp32/tools/move_ab.py --port COM5      # si no encuentra el puerto solo

Ctrl+C: el brazo se detiene en el acto y queda quieto con torque (no cae).

Ángulos en GRADOS, orden base, hombro, codo, pinza. Postura de arranque del brazo: 0,0,90,180
(hombro vertical, antebrazo horizontal). Límites (los aplica también el supervisor):
base ±91°, hombro -34..51°, codo 34..149°, pinza 92..183°.

Cada movimiento es una ACCIÓN: al salir su primer paso, el supervisor sube el pulso de sync
(GPIO25, activo en alto 1 ms, bajo en reposo) hacia el S3 y le manda un EVENT. Entre pulsos hay
al menos 1.5 s. Con --sin-pulso se mueve igual pero sin pulso.

Qué hace: reinicia el supervisor, comprueba que el brazo responde (posición real por UART),
simula el S3 en nivel 0 (sin S3 el supervisor no mueve nada), hace home y repite A -> B.
"""
import argparse
import math
import re
import sys
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit('Falta pyserial. Instálalo con:  python -m pip install pyserial')

INIT = (0.0, 0.0, 90.0, 180.0)
LIM = ((-91, 91), (-34, 51), (34, 149), (92, 183))
NAMES = ('base', 'hombro', 'codo', 'pinza')
MIN_SPACING_S = 1.5
# Chips USB-serie habituales en placas ESP32: WCH (CH340/CH9102), Silicon Labs (CP210x), FTDI, Espressif nativo.
USB_VIDS = {0x1A86, 0x10C4, 0x0403, 0x303A}


def find_port():
    ports = [p for p in list_ports.comports() if p.vid in USB_VIDS]
    if len(ports) == 1:
        return ports[0].device
    todos = '\n'.join(f'  {p.device}  {p.description}' for p in list_ports.comports()) or '  (ninguno)'
    if not ports:
        sys.exit('No encuentro el supervisor por USB. ¿Está conectado? ¿El cable es de datos?\n'
                 f'Puertos serie disponibles:\n{todos}\nSi es uno de ellos, usa --port <puerto>.')
    sys.exit(f'Hay varias placas conectadas; elige una con --port <puerto>:\n{todos}')


class Sup:
    def __init__(self, port, echo):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.05
        self.s.dtr = self.s.rts = False
        try:
            self.s.open()
        except serial.SerialException as e:
            hint = '\nEn Linux: añade tu usuario al grupo dialout (sudo usermod -aG dialout $USER) y vuelve a entrar.' \
                if sys.platform.startswith('linux') else ''
            sys.exit(f'No puedo abrir {port}: {e}{hint}\n¿Hay otro programa (monitor serie) usando el puerto?')
        self.lines = []
        self.lock = threading.Lock()
        self.echo = echo
        threading.Thread(target=self._read, daemon=True).start()
        self.s.rts = True; time.sleep(0.1); self.s.rts = False   # reinicio limpio del supervisor

    def _read(self):
        buf = b''
        while True:
            try:
                buf += self.s.read(512)
            except serial.SerialException:
                return
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode(errors='replace').strip()
                with self.lock:
                    self.lines.append(line)
                if self.echo and re.search(r' (sup|motion|safety|arm_uart|actions): ', line):
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
            for line in new:
                m = rx.search(line)
                if m:
                    return m
            time.sleep(0.02)
        return None

    def tail(self, start, n=8):
        with self.lock:
            return [l for l in self.lines[start:] if re.search(r' (sup|motion|safety|actions|arm_uart): ', l)][-n:]

    def real_pose(self):
        self.cmd('status')
        m = self.wait_for(r'posición REAL del brazo \(hace (\d+) ms\): b=([-+\d.]+) s=([-+\d.]+) e=([-+\d.]+) t=([-+\d.]+)', 2)
        if not m:
            return None
        return tuple(math.degrees(float(m.group(i))) for i in range(2, 6))


def parse_pose(text):
    try:
        v = tuple(float(x) for x in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('formato: base,hombro,codo,pinza en grados, p. ej. -30,0,90,180')
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
    pulse_txt = ''
    if pulse:
        if sup.wait_for(r'actions: PULSO acción', 0.5, start):
            _last_pulse[0] = time.time()
            pulse_txt = '  [pulso]'
        else:
            pulse_txt = '  [AVISO: sin pulso]'
    if dur > 0.05 and not sup.wait_for(r'motion: MOVING -> IDLE', dur + 3, start):
        print(f'{label}: AVISO, el movimiento no terminó en {dur + 3:.1f} s')
        for l in sup.tail(start):
            print('   ' + l)
    _last[:] = pose
    time.sleep(0.4)   # que el servo asiente y llegue una lectura nueva
    real = sup.real_pose()
    err = '' if real is None else '  error ' + ' '.join(f'{r - p:+.1f}' for r, p in zip(real, pose))
    print(f'{label}: real {fmt(real) if real else "(sin lectura)"}{err}{pulse_txt}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', type=parse_pose, default=(-30.0, 0.0, 90.0, 180.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--b', type=parse_pose, default=(30.0, 10.0, 100.0, 180.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--reps', type=int, default=0, help='idas y vueltas A->B->A; 0 = sin fin (por defecto)')
    ap.add_argument('--vel', type=float, default=20.0, help='velocidad máxima por articulación, °/s (máx 57)')
    ap.add_argument('--pausa', type=float, default=1.0, help='segundos quieto en A y en B')
    ap.add_argument('--sin-pulso', action='store_true', help='mover sin pulso de sync')
    ap.add_argument('-v', '--verbose', action='store_true', help='mostrar el log del supervisor')
    ap.add_argument('-p', '--port', help='puerto del supervisor (p. ej. /dev/ttyACM0, COM5); por defecto lo busca')
    a = ap.parse_args()
    a.vel = max(1.0, min(a.vel, 57.0))

    port = a.port or find_port()
    print(f'Supervisor en {port}. Reiniciándolo...', flush=True)
    sup = Sup(port, a.verbose)
    try:
        if not sup.wait_for(r'arm_uart: enlace UART', 8):
            raise SystemExit('el supervisor no arrancó o no tiene el firmware de INTUNE con enlace UART')
        time.sleep(1.0)
        real = sup.real_pose()
        if real is None:
            raise SystemExit('el brazo no responde por UART. ¿Está encendido (y terminó de arrancar)? '
                             '¿Cables: pin 10 del brazo -> RX2/GPIO16, pin 8 -> TX2/GPIO17, GND -> GND?')
        print(f'Brazo responde. Posición real: {fmt(real)}')

        sup.cmd('s3 0')
        if not sup.wait_for(r'-> RUN', 2):
            raise SystemExit('la seguridad del supervisor no pasó a RUN')
        sup.cmd(f'vel {a.vel}')
        print('Home (postura de arranque, lento)...', flush=True)
        start = sup.mark()
        sup.cmd('home')
        m = sup.wait_for(r'HOMING -> IDLE|home rechazado: (.*)', 10, start)
        if not m or m.group(1):
            raise SystemExit('home no terminó' + (f': {m.group(1)}' if m else '') + '\n  ' + '\n  '.join(sup.tail(start)))

        reps = f'{a.reps} idas y vueltas' if a.reps > 0 else 'sin fin'
        print(f'\nA = {fmt(a.a)}\nB = {fmt(a.b)}\n{reps} a {a.vel:.0f}°/s, pausa {a.pausa} s'
              f'{", sin pulso" if a.sin_pulso else ", con pulso de sync"}.  Ctrl+C para detener.\n', flush=True)
        i = 0
        while a.reps <= 0 or i < a.reps:
            i += 1
            go(sup, a.a, f'[{i}] A', a.vel, not a.sin_pulso)
            time.sleep(a.pausa)
            go(sup, a.b, f'[{i}] B', a.vel, not a.sin_pulso)
            time.sleep(a.pausa)

        print('\nVolviendo a la postura de arranque...', flush=True)
        go(sup, INIT, 'inicio', a.vel)
    except KeyboardInterrupt:
        sup.cmd('stop')        # frena en el último paso
        print('\nCtrl+C: brazo detenido.', flush=True)
        time.sleep(0.3)
    finally:
        try:
            sup.cmd('s3 off')  # sin heartbeat el supervisor retiene el brazo (quieto, con torque)
            time.sleep(0.5)
            sup.s.close()
        except Exception:
            pass
        print('Brazo retenido y quieto (con torque).')


if __name__ == '__main__':
    main()
