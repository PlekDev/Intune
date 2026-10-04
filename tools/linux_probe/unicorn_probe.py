#!/usr/bin/env python3
"""
Prueba 1 (Linux, sin ESP): conectar al Unicorn Hybrid Black por RFCOMM, enviar
start, parsear tramas de 45 bytes y medir integridad (contador, tasa, ráfagas).

Dos transportes:
  * Socket RFCOMM nativo de Python (por defecto, no requiere rfcomm/pyserial):
        python3 unicorn_probe.py AA:BB:CC:DD:EE:FF --channel 1
  * Puerto serie ya enlazado (rfcomm bind 0 <MAC> <canal>), requiere pyserial:
        python3 unicorn_probe.py --port /dev/rfcomm0

Antes: emparejar con bluetoothctl (power on / agent on / default-agent /
scan on / pair <MAC> / trust <MAC>) y ANOTAR si pidió PIN.
Si no se conoce el canal RFCOMM (sdptool ya no viene en BlueZ moderno):
        python3 unicorn_probe.py <MAC> --find-channel

Otras opciones útiles:
  --hex N        volcar en hex los primeros N bloques recibidos (crudo)
  --csv f.csv    guardar muestras (contador, batería, 8 EEG µV, acc, gyro, t_llegada)
  --signal       cada segundo: RMS y potencia relativa alfa (8–12 Hz) por canal
  --duration S   cortar a los S segundos (Ctrl+C también corta y envía stop)
  --selftest     probar el parser con tramas sintéticas (sin hardware)
"""

import argparse
import math
import socket
import struct
import sys
import time
from collections import Counter

# ---- Protocolo (ver CLAUDE.md) ----
CMD_START = bytes([0x61, 0x7C, 0x87])
CMD_STOP = bytes([0x63, 0x5C, 0xC5])
ACK = bytes([0x00, 0x00, 0x00])
FRAME_LEN = 45
HEADER = b"\xC0\x00"
FOOTER = b"\x0D\x0A"
FS = 250.0
N_EEG = 8
EEG_SCALE = 4500000.0 / 50331642.0  # raw -> µV
ACC_SCALE = 1.0 / 4096.0  # -> g
GYR_SCALE = 1.0 / 32.8  # -> °/s
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]  # verificar en el manual


def decode_frame(f):
    """f: 45 bytes ya validados (cabecera/terminador). Devuelve dict."""
    battery = 100.0 * (f[2] & 0x0F) / 15.0
    eeg = []
    for ch in range(N_EEG):
        o = 3 + 3 * ch
        raw = int.from_bytes(f[o:o + 3], "big", signed=True)  # extensión de signo correcta
        eeg.append(raw * EEG_SCALE)
    acc = [v * ACC_SCALE for v in struct.unpack_from("<3h", f, 27)]
    gyr = [v * GYR_SCALE for v in struct.unpack_from("<3h", f, 33)]  # 33:39, no 27:33
    (counter,) = struct.unpack_from("<I", f, 39)
    return {"battery": battery, "eeg": eeg, "acc": acc, "gyr": gyr, "counter": counter}


class Parser:
    """Parser con resincronización: busca C0 00 y exige 0D 0A en +43."""

    def __init__(self):
        self.buf = bytearray()
        self.discarded = 0  # bytes descartados durante resync

    def feed(self, data):
        self.buf += data
        out = []
        b = self.buf
        i = 0
        n = len(b)
        while n - i >= FRAME_LEN:
            if b[i] == 0xC0 and b[i + 1] == 0x00 and b[i + 43] == 0x0D and b[i + 44] == 0x0A:
                out.append(bytes(b[i:i + FRAME_LEN]))
                i += FRAME_LEN
            else:
                i += 1
                self.discarded += 1
        del b[:i]
        return out


class Stats:
    def __init__(self):
        self.t0 = time.monotonic()
        self.bytes = 0
        self.frames = 0
        self.gaps = 0  # eventos de hueco
        self.lost = 0  # muestras faltantes
        self.backwards = 0  # contador que retrocede o se repite
        self.prev_cnt = None
        self.first_cnt = None
        self.battery = None
        self.chunk_dt = Counter()  # histograma de inter-arrival de lecturas (ms, bins de 5)
        self.chunk_sizes = Counter()
        self.frames_per_read = Counter()
        self.last_read_t = None
        # ventana por segundo
        self.win_frames = 0
        self.win_bytes = 0
        self.win_t = self.t0

    def on_read(self, data, t, nframes):
        self.bytes += len(data)
        self.win_bytes += len(data)
        if self.last_read_t is not None:
            dt_ms = (t - self.last_read_t) * 1000.0
            self.chunk_dt[min(int(dt_ms // 5) * 5, 200)] += 1
        self.last_read_t = t
        self.chunk_sizes[len(data)] += 1
        self.frames_per_read[nframes] += 1

    def on_frame(self, d):
        self.frames += 1
        self.win_frames += 1
        self.battery = d["battery"]
        c = d["counter"]
        if self.prev_cnt is None:
            self.first_cnt = c
        else:
            diff = (c - self.prev_cnt) & 0xFFFFFFFF
            if diff == 1:
                pass
            elif diff == 0 or diff > 0x80000000:
                self.backwards += 1
            else:
                self.gaps += 1
                self.lost += diff - 1
                print(f"  [hueco] {self.prev_cnt} -> {c}  ({diff - 1} muestras perdidas)")
        self.prev_cnt = c

    def loss_pct(self):
        expected = self.frames + self.lost
        return 100.0 * self.lost / expected if expected else 0.0


class SignalMonitor:
    """RMS y potencia relativa alfa por canal en ventanas de 1 s (DFT directa, sin numpy)."""

    def __init__(self, n=int(FS)):
        self.n = n
        self.win = [[] for _ in range(N_EEG)]
        # tablas de cos/sin para bins 1..40 Hz (resolución 1 Hz con n=250)
        self.bins = list(range(1, 41))
        self.tab = {k: ([math.cos(2 * math.pi * k * t / n) for t in range(n)],
                        [math.sin(2 * math.pi * k * t / n) for t in range(n)]) for k in self.bins}

    def push(self, eeg):
        for ch, v in enumerate(eeg):
            self.win[ch].append(v)
        if len(self.win[0]) >= self.n:
            self.report()
            self.win = [[] for _ in range(N_EEG)]

    def report(self):
        parts = []
        for ch in range(N_EEG):
            x = self.win[ch]
            m = sum(x) / len(x)
            x = [v - m for v in x]
            rms = math.sqrt(sum(v * v for v in x) / len(x))
            p = {}
            for k in self.bins:
                c, s = self.tab[k]
                re = sum(a * b for a, b in zip(x, c))
                im = sum(a * b for a, b in zip(x, s))
                p[k] = re * re + im * im
            # alfa / banda 2–40 Hz (excluye deriva lenta y DC)
            tot = sum(p[k] for k in self.bins if k >= 2) or 1.0
            alpha = sum(p[k] for k in range(8, 13)) / tot
            parts.append(f"{CHANNELS[ch]}:{rms:7.1f}µV α{100 * alpha:4.0f}%")
        print("  [señal] " + " | ".join(parts))


# ---- Transportes ----

class RfcommSocket:
    def __init__(self, mac, channel, timeout=1.0):
        self.s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self.s.settimeout(10.0)
        self.s.connect((mac, channel))
        self.s.settimeout(timeout)

    def read(self, n=4096):
        try:
            return self.s.recv(n)
        except socket.timeout:
            return b""

    def write(self, data):
        self.s.sendall(data)

    def close(self):
        self.s.close()


class SerialPort:
    def __init__(self, port, timeout=1.0):
        try:
            import serial
        except ImportError:
            sys.exit("Falta pyserial: pip install pyserial (o usar el modo socket con la MAC)")
        self.s = serial.Serial(port, 115200, timeout=timeout)

    def read(self, n=4096):
        waiting = self.s.in_waiting
        return self.s.read(max(1, min(n, waiting)) if waiting else 1)

    def write(self, data):
        self.s.write(data)
        self.s.flush()

    def close(self):
        self.s.close()


def find_channel(mac):
    """Sin sdptool: probar conectar a canales RFCOMM 1..30."""
    found = []
    for ch in range(1, 31):
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        s.settimeout(8.0)
        try:
            s.connect((mac, ch))
            print(f"  canal {ch}: CONECTA")
            found.append(ch)
        except OSError as e:
            print(f"  canal {ch}: {e.strerror or e}")
        finally:
            s.close()
        time.sleep(0.3)
    print(f"Canales que aceptan conexión: {found or 'ninguno'}")


# ---- Flujo principal ----

def wait_ack(link, timeout=3.0):
    """Lee hasta encontrar el ACK 00 00 00. Devuelve (ok, bytes_sobrantes_tras_ack)."""
    buf = bytearray()
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        buf += link.read()
        i = buf.find(ACK)
        if i >= 0:
            if i:
                print(f"  {i} bytes antes del ACK: {bytes(buf[:i]).hex(' ')}")
            return True, bytes(buf[i + 3:])
    print(f"  sin ACK; recibido: {bytes(buf[:64]).hex(' ')}{' ...' if len(buf) > 64 else ''}")
    return False, bytes(buf)


def run(args):
    if args.port:
        print(f"Abriendo {args.port} ...")
        link = SerialPort(args.port)
    else:
        print(f"Conectando RFCOMM a {args.mac} canal {args.channel} ...")
        link = RfcommSocket(args.mac, args.channel)
    print("Conectado.")

    csv = open(args.csv, "w") if args.csv else None
    if csv:
        csv.write("counter,battery," + ",".join(f"{c}_uV" for c in CHANNELS)
                  + ",acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,t_arrival_s\n")

    parser = Parser()
    stats = Stats()
    sig = SignalMonitor() if args.signal else None
    hex_left = args.hex
    printed_first = False

    try:
        if not args.no_start:
            # por si quedó transmitiendo de una sesión anterior
            link.write(CMD_STOP)
            time.sleep(0.2)
            while link.read():
                pass
            print(f"Enviando start {CMD_START.hex(' ')} ...")
            link.write(CMD_START)
            ok, rest = wait_ack(link)
            print("ACK OK" if ok else "ACK NO recibido (sigo leyendo de todas formas)")
        else:
            rest = b""

        stats = Stats()
        t_end = stats.t0 + args.duration if args.duration else None
        pending = rest
        while True:
            now = time.monotonic()
            if t_end and now >= t_end:
                break
            data = pending or link.read()
            pending = b""
            t = time.monotonic()
            if data:
                if hex_left > 0:
                    print(f"  [raw {len(data):4d} B] {data.hex(' ')}")
                    hex_left -= 1
                frames = parser.feed(data)
                stats.on_read(data, t, len(frames))
                for f in frames:
                    d = decode_frame(f)
                    stats.on_frame(d)
                    if not printed_first:
                        printed_first = True
                        print(f"  Primera trama: {f.hex(' ')}")
                        print(f"    cnt={d['counter']} bat={d['battery']:.0f}% "
                              f"eeg=[{', '.join(f'{v:.1f}' for v in d['eeg'])}] µV "
                              f"acc=[{', '.join(f'{v:.2f}' for v in d['acc'])}] g "
                              f"gyr=[{', '.join(f'{v:.1f}' for v in d['gyr'])}] °/s")
                    if csv:
                        csv.write(f"{d['counter']},{d['battery']:.1f},"
                                  + ",".join(f"{v:.3f}" for v in d["eeg"]) + ","
                                  + ",".join(f"{v:.4f}" for v in d["acc"]) + ","
                                  + ",".join(f"{v:.3f}" for v in d["gyr"])
                                  + f",{t - stats.t0:.6f}\n")
                    if sig:
                        sig.push(d["eeg"])

            if t - stats.win_t >= 1.0:
                dt = t - stats.win_t
                print(f"[{t - stats.t0:6.1f}s] {stats.win_frames / dt:6.1f} tramas/s  "
                      f"{stats.win_bytes / dt / 1000:5.2f} kB/s  total={stats.frames}  "
                      f"huecos={stats.gaps} perdidas={stats.lost} ({stats.loss_pct():.3f}%)  "
                      f"descartados={parser.discarded}  bat={stats.battery if stats.battery is None else round(stats.battery)}%")
                stats.win_frames = 0
                stats.win_bytes = 0
                stats.win_t = t
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        try:
            if not args.no_start:
                print(f"Enviando stop {CMD_STOP.hex(' ')} ...")
                link.write(CMD_STOP)
                time.sleep(0.3)
        except OSError as e:
            print(f"  error enviando stop: {e}")
        link.close()
        if csv:
            csv.close()
        summary(stats, parser)


def summary(stats, parser):
    el = time.monotonic() - stats.t0
    print("\n===== Resumen =====")
    print(f"Duración: {el:.1f} s   bytes: {stats.bytes}   tramas: {stats.frames}")
    if el > 0:
        print(f"Tasa media: {stats.frames / el:.2f} tramas/s (esperado {FS:.0f})")
    if stats.first_cnt is not None:
        span = (stats.prev_cnt - stats.first_cnt) & 0xFFFFFFFF
        print(f"Contador: {stats.first_cnt} -> {stats.prev_cnt} (span {span + 1}, "
              f"→ {(span + 1) / FS:.1f} s de muestras)")
    print(f"Huecos: {stats.gaps}  muestras perdidas: {stats.lost}  pérdida: {stats.loss_pct():.4f}%  "
          f"(criterio prueba 5: < 0.1%) -> {'PASA' if stats.frames and stats.loss_pct() < 0.1 else 'NO PASA'}")
    print(f"Contador repetido/retrocede: {stats.backwards}   bytes descartados en resync: {parser.discarded}")
    if stats.chunk_dt:
        print("Inter-arrival entre lecturas (ms):")
        total = sum(stats.chunk_dt.values())
        for k in sorted(stats.chunk_dt):
            n = stats.chunk_dt[k]
            label = f"{k:3d}-{k + 5:<3d}" if k < 200 else "≥200   "
            print(f"  {label} {n:7d} {'#' * max(1, int(50 * n / total)) if n else ''}")
    if stats.frames_per_read:
        top = sorted(stats.frames_per_read.items())
        print("Tramas completas por lectura: " + ", ".join(f"{k}:{v}" for k, v in top))


# ---- Autotest del parser ----

def make_frame(counter, eeg_raw, acc=(0, 0, 4096), gyr=(0, 0, 0), bat=0x0F):
    f = bytearray(HEADER)
    f.append(bat)
    for r in eeg_raw:
        f += (r & 0xFFFFFF).to_bytes(3, "big")
    f += struct.pack("<3h", *acc)
    f += struct.pack("<3h", *gyr)
    f += struct.pack("<I", counter)
    f += FOOTER
    assert len(f) == FRAME_LEN
    return bytes(f)


def selftest():
    eeg = [-1, 1, -8388608, 8388607, 0, -1000, 1000, 4194304]
    stream = b"\x00\x00\x00"  # ACK
    stream += b"\xC0\x00\x0D\x0A\x99"  # basura que parece cabecera
    counters = [10, 11, 12, 15, 16]  # hueco de 2
    for c in counters:
        stream += make_frame(c, eeg, gyr=(328, -328, 0))
    p, st = Parser(), Stats()
    # alimentar en trozos irregulares para simular ráfagas BT
    frames = []
    for i in range(0, len(stream), 17):
        frames += p.feed(stream[i:i + 17])
    for f in frames:
        st.on_frame(decode_frame(f))
    d = decode_frame(frames[0])
    assert [round(v / EEG_SCALE) for v in d["eeg"]] == eeg, d["eeg"]
    assert abs(d["eeg"][0] + EEG_SCALE) < 1e-9  # -1 raw ≈ -0.089 µV, no ~+380 V
    assert abs(d["gyr"][0] - 10.0) < 1e-6 and abs(d["gyr"][1] + 10.0) < 1e-6
    assert abs(d["acc"][2] - 1.0) < 1e-9
    assert d["battery"] == 100.0
    assert len(frames) == 5 and st.gaps == 1 and st.lost == 2, (len(frames), st.gaps, st.lost)
    assert p.discarded == 8, p.discarded
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mac", nargs="?", help="MAC del Unicorn (UN-xxxx)")
    ap.add_argument("--channel", type=int, default=1, help="canal RFCOMM (def. 1)")
    ap.add_argument("--port", help="usar puerto serie (p.ej. /dev/rfcomm0) en vez de socket")
    ap.add_argument("--find-channel", action="store_true", help="probar canales RFCOMM 1..30")
    ap.add_argument("--duration", type=float, default=0, help="segundos (0 = hasta Ctrl+C)")
    ap.add_argument("--hex", type=int, default=0, help="volcar en hex los primeros N bloques")
    ap.add_argument("--csv", help="guardar muestras decodificadas")
    ap.add_argument("--signal", action="store_true", help="RMS y alfa relativa por canal cada 1 s")
    ap.add_argument("--no-start", action="store_true", help="no enviar start/stop (solo escuchar)")
    ap.add_argument("--selftest", action="store_true", help="probar el parser sin hardware")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.find_channel:
        if not args.mac:
            ap.error("--find-channel requiere la MAC")
        find_channel(args.mac)
        return
    if not args.mac and not args.port:
        ap.error("indicar MAC o --port")
    run(args)


if __name__ == "__main__":
    main()
