#!/usr/bin/env python3
"""INTUNE C3: mueve el brazo por tres puntos A -> B -> C -> A ..., sin fin, a través del supervisor.

Uso (Linux, macOS o Windows; solo necesita pyserial):
  python firmware/arm_esp32/tools/move_abc.py                  # puntos por defecto, hasta Ctrl+C
  python firmware/arm_esp32/tools/move_abc.py --vel 30 --pausa 1.5
  python firmware/arm_esp32/tools/move_abc.py --a -70,0,90,180 --b 0,25,115,130 --c 70,-10,80,180
  python firmware/arm_esp32/tools/move_abc.py --reps 2 --port COM5

Ctrl+C: el brazo se detiene en el acto y queda quieto con torque (no cae).

Puntos por defecto (grados: base, hombro, codo, pinza; arranque = 0,0,90,180):
  A  -60,   0,  90, 180   girado a un lado, postura de arranque
  B    0,  20, 110, 130   al centro, inclinado hacia delante y abajo, pinza abierta
  C   60, -10,  80, 180   girado al otro lado, levantado hacia atrás
Límites: base ±91°, hombro -34..51°, codo 34..149°, pinza 92..183°.

Cada movimiento es una ACCIÓN con pulso de sync al S3 (GPIO25, alto 1 ms) y EVENT; --sin-pulso
para moverlo sin pulsos. Necesita move_ab.py en la misma carpeta (comparten la lógica).
"""
import argparse

from move_ab import add_common_args, parse_pose, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', type=parse_pose, default=(-60.0, 0.0, 90.0, 180.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--b', type=parse_pose, default=(0.0, 20.0, 110.0, 130.0), help='grados: base,hombro,codo,pinza')
    ap.add_argument('--c', type=parse_pose, default=(60.0, -10.0, 80.0, 180.0), help='grados: base,hombro,codo,pinza')
    add_common_args(ap)
    a = ap.parse_args()
    run([('A', a.a), ('B', a.b), ('C', a.c)], a)


if __name__ == '__main__':
    main()
