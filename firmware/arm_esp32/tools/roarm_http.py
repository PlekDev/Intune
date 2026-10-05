#!/usr/bin/env python3
"""INTUNE C3: pruebas del RoArm-M2-S por HTTP (firmware de fábrica, AP "RoArm-M2").

Envía comandos JSON a http://<host>/js?json=... y mide la latencia de ida y vuelta.
Solo librería estándar: funciona sin internet mientras el PC está en la red del brazo.

Modos:
  roarm_http.py --suite            # batería de pruebas guiada, todo queda en logs/roarm_<fecha>.log
  roarm_http.py --poll 20          # lee posición (T:105) 20 veces y resume latencias
  roarm_http.py '{"T":105}'        # un comando cualquiera

La suite solo mueve la BASE (giro horizontal, sin carga de gravedad) a velocidad baja,
y pide Enter antes de cada prueba con movimiento. Ctrl+C en cualquier momento: manda
como objetivo la posición actual (el brazo se queda quieto CON torque).

{"T":0} está bloqueado: el paro de fábrica quita el torque 10 s y el brazo cae.
"""
import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request

BLOCKED = {0: 'T:0 quita el torque a todos los servos durante 10 s (el brazo cae). Usa --force si de verdad lo quieres.'}
BASE = 1            # BASE_JOINT en RoArm-M2_config.h
SPD_SAFE = 200      # pasos/s del servo (4096 pasos = 1 vuelta) ~ 17.6 °/s. spd=0 es velocidad MÁXIMA.
ACC_SAFE = 10
TOL = 0.03          # rad (~1.7°) para considerar que llegó

HOST = '192.168.4.1'
TIMEOUT = 3.0
_log = None


def out(msg=''):
    line = f'{datetime.datetime.now():%H:%M:%S.%f}'[:-3] + '  ' + msg
    print(line, flush=True)
    if _log:
        _log.write(line + '\n')
        _log.flush()


def send(cmd):
    url = f'http://{HOST}/js?json=' + urllib.parse.quote(json.dumps(cmd, separators=(',', ':')))
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        body = r.read().decode(errors='replace')
    return body, (time.perf_counter() - t0) * 1000


def read_pos():
    """T:105 -> (dict de feedback, latencia ms)."""
    body, ms = send({'T': 105})
    return json.loads(body), ms


def move_base(rad, spd, acc=ACC_SAFE):
    assert spd > 0, 'spd=0 es velocidad máxima'
    body, ms = send({'T': 101, 'joint': BASE, 'rad': round(rad, 4), 'spd': spd, 'acc': acc})
    out(f'  -> T:101 base={rad:+.3f} rad spd={spd} acc={acc}  ({ms:.0f} ms)')
    return ms


def hold_here():
    """Objetivo = posición actual de la base: se queda quieta con torque."""
    fb, _ = read_pos()
    send({'T': 101, 'joint': BASE, 'rad': fb['b'], 'spd': SPD_SAFE, 'acc': 0})
    return fb['b']


def track(target, max_s, every=0.1):
    """Lee la base hasta llegar a target (±TOL) o agotar max_s. Devuelve (llegó, segundos, último b)."""
    t0 = time.perf_counter()
    b = None
    while True:
        fb, ms = read_pos()
        b = fb['b']
        el = time.perf_counter() - t0
        out(f'     t={el:5.2f}s  b={b:+.3f}  ({ms:.0f} ms)')
        if target is not None and abs(b - target) < TOL:
            return True, el, b
        if el > max_s:
            return False, el, b
        time.sleep(every)


def ask(title, what):
    out()
    out(f'=== {title} ===')
    out(what)
    r = input('   Enter = hacer   s = saltar   q = terminar > ').strip().lower()
    out(f'   respuesta: {r or "enter"}')
    if r == 'q':
        raise SystemExit('terminado por el usuario')
    return r != 's'


def suite():
    out(f'INTUNE C3 suite RoArm-M2-S, host {HOST}')

    # 0. Conexión
    try:
        fb, ms = read_pos()
    except OSError as e:
        out(f'SIN CONEXIÓN con {HOST}: {e}')
        out('¿El PC está conectado a la red Wi-Fi "RoArm-M2"? ¿El brazo está encendido?')
        return
    out(f'conectado ({ms:.0f} ms): {json.dumps(fb)}')

    # 1. Solo lectura
    out()
    out('=== Prueba 1: lectura de posición, 30 veces (NO se mueve) ===')
    lat = []
    for i in range(30):
        try:
            fb, ms = read_pos()
            lat.append(ms)
            out(f'  {i:2d} {ms:6.1f} ms  b={fb["b"]:+.3f} s={fb["s"]:+.3f} e={fb["e"]:+.3f} t={fb["t"]:+.3f}')
        except (OSError, ValueError, KeyError) as e:
            out(f'  {i:2d} ERROR {e}')
    if lat:
        out(f'RESUMEN P1: ok {len(lat)}/30  min {min(lat):.1f}  mediana {statistics.median(lat):.1f}  max {max(lat):.1f} ms')

    b0 = read_pos()[0]['b']
    sign = -1 if b0 > 0 else 1      # girar hacia el centro, lejos del límite ±pi
    out(f'base inicial b0={b0:+.3f} rad; las pruebas giran hacia {"-" if sign < 0 else "+"}')

    # 2. Movimiento pequeño y comprobación del signo
    if ask('Prueba 2: giro pequeño de la base',
           f'Girará la base ~17° (0.3 rad) a {SPD_SAFE} pasos/s (~18°/s) y volverá. Hombro, codo y pinza no se mueven.'):
        tgt = b0 + sign * 0.3
        move_base(tgt, SPD_SAFE)
        ok, el, b = track(tgt, 5)
        out(f'RESUMEN P2 ida: {"llegó" if ok else "NO llegó"} en {el:.2f} s, b={b:+.3f}, objetivo {tgt:+.3f}')
        if not ok and abs(b - tgt) > 0.1:
            out('ALERTA: la lectura no coincide con el objetivo (¿signo/convención distinta?). '
                'Detengo las pruebas con movimiento por seguridad.')
            hold_here()
            return
        move_base(b0, SPD_SAFE)
        ok, el, b = track(b0, 5)
        out(f'RESUMEN P2 vuelta: {"llegó" if ok else "NO llegó"} en {el:.2f} s, b={b:+.3f}')
    else:
        out('Prueba 2 saltada: no se harán las pruebas 3 y 4 (dependen de verificar el signo).')
        return

    # 3. Velocidades
    if ask('Prueba 3: velocidades',
           'Girará la base ~29° (0.5 rad) ida y vuelta a 300, 800 y 1500 pasos/s (~26, 70 y 132 °/s).'):
        for spd in (300, 800, 1500):
            for tgt in (b0 + sign * 0.5, b0):
                t_cmd = move_base(tgt, spd)
                ok, el, b = track(tgt, 6, every=0.05)
                deg_s = math.degrees(0.5) / el if ok and el > 0 else float('nan')
                out(f'RESUMEN P3 spd={spd}: {"llegó" if ok else "NO llegó"} en {el:.2f} s '
                    f'(~{deg_s:.0f} °/s medido, comando {t_cmd:.0f} ms)')
            time.sleep(0.5)

    # 4. Parar a mitad de movimiento manteniendo torque
    if ask('Prueba 4: parada a mitad de movimiento',
           'Iniciará un giro de ~57° (1.0 rad) a 300 pasos/s y a los 0.8 s mandará "quédate donde estás" '
           '(objetivo = posición actual, con torque). Debe frenar y quedarse firme 3 s; luego vuelve.'):
        move_base(b0 + sign * 1.0, 300)
        time.sleep(0.8)
        fb, ms_r = read_pos()
        b_cmd = fb['b']
        t0 = time.perf_counter()
        _, ms_h = send({'T': 101, 'joint': BASE, 'rad': b_cmd, 'spd': 300, 'acc': 0})
        out(f'  -> HOLD en b={b_cmd:+.3f} (lectura {ms_r:.0f} ms + comando {ms_h:.0f} ms)')
        _, _, b_end = track(None, 3)
        out(f'RESUMEN P4: pidió parar en {b_cmd:+.3f}, quedó en {b_end:+.3f} '
            f'(sobrepaso {math.degrees(abs(b_end - b_cmd)):.1f}°), tiempo hold {(time.perf_counter() - t0):.1f} s')
        out('   ¿Se quedó firme? Anota si al empujarlo suavemente con la mano resiste.')
        input('   Enter para volver a la posición inicial > ')
        move_base(b0, SPD_SAFE)
        track(b0, 8)

    out()
    out('=== Suite terminada. Comparte este log con C3. ===')


def main():
    global HOST, TIMEOUT, _log
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', nargs='?', help='comando JSON')
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--suite', action='store_true', help='batería de pruebas guiada con log')
    ap.add_argument('--poll', type=int, metavar='N', help='lee T:105 N veces y resume latencias')
    ap.add_argument('--timeout', type=float, default=TIMEOUT)
    ap.add_argument('--force', action='store_true', help='permite comandos bloqueados')
    a = ap.parse_args()
    HOST, TIMEOUT = a.host, a.timeout

    if a.suite:
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(logdir, exist_ok=True)
        path = os.path.join(logdir, f'roarm_{datetime.datetime.now():%Y%m%d_%H%M%S}.log')
        _log = open(path, 'w')
        print(f'log: {path}')
        try:
            suite()
        except KeyboardInterrupt:
            out('Ctrl+C: mando HOLD (posición actual, con torque)')
            try:
                out(f'HOLD en b={hold_here():+.3f}')
            except Exception as e:
                out(f'HOLD falló: {e}. Usa el interruptor de corriente si hace falta.')
        except Exception as e:
            out(f'ERROR: {type(e).__name__}: {e}')
        finally:
            print(f'\nlog guardado en: {path}')
        return

    if a.poll:
        lat = []
        for i in range(a.poll):
            try:
                body, ms = send({'T': 105})
                lat.append(ms)
                print(f'{i:3d} {ms:7.1f} ms  {body}')
            except OSError as e:
                print(f'{i:3d} ERROR {e}')
        if lat:
            print(f'\nok {len(lat)}/{a.poll}  min {min(lat):.1f}  mediana {statistics.median(lat):.1f}  max {max(lat):.1f} ms')
        return

    if not a.cmd:
        ap.error('falta el comando JSON (o usa --suite / --poll N)')
    cmd = json.loads(a.cmd)
    if cmd.get('T') in BLOCKED and not a.force:
        sys.exit(f'bloqueado: {BLOCKED[cmd["T"]]}')
    body, ms = send(cmd)
    print(f'{ms:.1f} ms  {body}')


if __name__ == '__main__':
    main()
