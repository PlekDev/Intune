#!/usr/bin/env python3
"""INTUNE C3: manda al supervisor una secuencia 'comando@espera_s' y muestra su salida.

  sup_seq.py 'home@6' 'gob 0.8@3' 'status@0.5'
"""
import sys
import time

import serial
from sup_console import find_port

s = serial.Serial()
s.port, s.baudrate, s.timeout = find_port(), 115200, 0.05
s.dtr = s.rts = False
s.open()
s.rts = True; time.sleep(0.1); s.rts = False


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        d = s.read(512)
        if d:
            sys.stdout.write(d.decode(errors='replace'))
            sys.stdout.flush()


pump(5)   # arranque + scan
for item in sys.argv[1:]:
    cmd, _, wait = item.rpartition('@')
    print(f'\n>>> {cmd}', flush=True)
    s.write(cmd.encode() + b'\n')
    pump(float(wait))
