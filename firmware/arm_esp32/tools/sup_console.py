#!/usr/bin/env python3
"""INTUNE C3: consola del supervisor por USB.

  sup_console.py                         # muestra la salida 8 s (abrir el puerto reinicia la placa)
  sup_console.py -s 3 'status' 'base 0.2'  # espera 3 s tras el arranque y manda comandos
  sup_console.py -i                      # interactivo: escribe comandos, Ctrl+C para salir
"""
import argparse
import glob
import sys
import threading
import time

import serial


def find_port():
    ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    if not ports:
        sys.exit('no hay /dev/ttyACM* ni /dev/ttyUSB*')
    return ports[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmds', nargs='*', help='comandos a enviar en orden')
    ap.add_argument('-p', '--port')
    ap.add_argument('-s', '--settle', type=float, default=6.0, help='segundos de espera tras el arranque')
    ap.add_argument('-g', '--gap', type=float, default=1.5, help='segundos entre comandos')
    ap.add_argument('-t', '--tail', type=float, default=2.0, help='segundos de salida tras el último comando')
    ap.add_argument('-i', '--interactive', action='store_true')
    a = ap.parse_args()

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = a.port or find_port(), 115200, 0.1
    s.dtr = s.rts = False          # si quedan activas, el auto-reset deja la placa en reset
    s.open()
    s.rts = True; time.sleep(0.1); s.rts = False   # un reset limpio para ver el arranque
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            data = s.read(512)
            if data:
                sys.stdout.write(data.decode(errors='replace'))
                sys.stdout.flush()

    threading.Thread(target=reader, daemon=True).start()
    time.sleep(a.settle)
    try:
        if a.interactive:
            for line in sys.stdin:
                s.write(line.strip().encode() + b'\n')
        else:
            for c in a.cmds:
                print(f'\n>>> {c}', flush=True)
                s.write(c.encode() + b'\n')
                time.sleep(a.gap)
            time.sleep(a.tail)
    except KeyboardInterrupt:
        pass
    stop.set()


if __name__ == '__main__':
    main()
