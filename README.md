# INTUNE

**I**ntegrated **N**eural **T**elemetry for **U**nsafe-state **N**otification and **E**mergency response

*Keeping robots in tune with the people who operate them.*

BR41N.IO Hackathon project (IEEE SMC 2026, Programming category: "Your Hacking Project").

A fully embedded pipeline (no PC, no g.tec dongle) that monitors an operator's EEG while they work with a robotic arm. It detects **error-related potentials (ErrPs)**, the brain response that appears when the operator sees the arm do something wrong, and sends **graded alerts** to the arm so it can react, for example slowing down on a mild alert or stopping on a severe one. The target is industrial use, but the design is general purpose: any setting where a person operates or supervises a robotic arm.

> Hackathon prototype. Not a certified safety system.

## Architecture

```
Unicorn Hybrid Black ──BT Classic SPP──> classic ESP32 ──UART 921600──> ESP32-S3 <────UART─────> arm ESP32
   8 EEG ch @ 250 Hz                     receives, validates,           ErrP detection   <──GPIO── applies the reaction
                                         causal IIR 1–15 Hz             (autoencoder),   sync     (slow down, pause, stop),
                                                                        graded alerts    pulse    sends event metadata
```

| Node | Role |
|---|---|
| **Classic ESP32** | Connects to the g.tec Unicorn Hybrid Black over Bluetooth Classic (the S3 has BLE only). Parses and validates frames, filters the EEG with a causal **1–15 Hz IIR band-pass** per channel and streams it to the S3. |
| **ESP32-S3** | Cuts the filtered EEG into epochs time-locked to each arm action, detects ErrPs with an **autoencoder** and turns the result into an alert level. |
| **Arm ESP32** | Controls the robotic arm. Fires a GPIO sync pulse to the S3 at the onset of each action (plus its metadata over UART), receives alerts over UART and applies the matching reaction. |

### Why ErrPs

When a person sees a machine make a mistake, the EEG shows an error-related potential: a fronto-central negativity about 200–300 ms after the event, followed by a positivity. The Unicorn's Fz and Cz electrodes sit right where it is strongest. Detecting it lets the arm react to the operator noticing a problem, sometimes before they press any button.

### Alert levels (draft)

| Level | Meaning | Arm reaction |
|---|---|---|
| 0 | Normal | Nominal speed |
| 1 | Mild | Slows down |
| 2 | Moderate | Pauses / asks for confirmation |
| 3 | Severe | Safe stop |

The system is fail-safe: if the arm loses the heartbeat from the S3, or the S3 loses valid EEG, the arm goes to a safe state.

## Status

| Step | Status |
|---|---|
| PC ↔ Unicorn over RFCOMM (protocol verified) | ✅ 0 % loss at 250 Hz |
| ESP32 ↔ Unicorn: discovery, connection, start/ACK | ✅ |
| Frame validation on the ESP32 | ✅ 11,330 consecutive frames, 0 gaps |
| 10-minute integrity test on the ESP32 | ✅ 0.001 % loss (2 of 176,197 samples, 1 gap), 0 corrupt frames, 0 reconnections in 11.5 min |
| Signal check (alpha with eyes closed, jaw artifact) | ✅ Through the ESP32 with the 1–15 Hz IIR (`link_view.py live`): alpha peak with eyes closed, large jaw-clench artifact, clean Fz/Cz |
| IIR filter + gap handling on the classic ESP32 | ⏳ |
| UART link ESP32 → S3 | ⏳ |
| Autoencoder on the S3 | ⏳ |
| UART link S3 → arm, arm controller | ⏳ |

Findings so far:
- **No pairing needed.** The Unicorn accepts an RFCOMM connection on channel 1 with no PIN or SSP.
- **Radio range matters.** The PCB antenna of a WROOM-32 has noticeably less range than a laptop adapter. Keep the ESP32 close to the operator, or use a WROOM-32U with an external antenna.

## Repository layout

```
tools/linux_probe/unicorn_probe.py   # PC tool: connect to the Unicorn, parse, measure loss, CSV export
firmware/common/                     # Shared headers (Unicorn protocol + parser, inter-node protocols)
firmware/bridge_esp32/               # Classic ESP32: BT acquisition + IIR filter (PlatformIO, ESP-IDF)
firmware/detector_s3/                # ESP32-S3: autoencoder + alerts
firmware/arm_esp32/                  # Arm controller: UART from the S3 + reactions
ml/                                  # Autoencoder training and data
```

## Getting started

### Requirements

- Linux with BlueZ (tested with Python 3.14, standard library only).
- [PlatformIO](https://platformio.org/). We install it with `uv tool install --python 3.12 platformio`.
- g.tec Unicorn Hybrid Black, a classic ESP32 (e.g. ESP32-DevKitC / WROOM-32) and an ESP32-S3.

### 1. Check the headset from a PC

```bash
cd tools/linux_probe
python3 unicorn_probe.py --selftest                        # parser self-test, no hardware
python3 unicorn_probe.py <MAC> --hex 3 --duration 30       # connect, start, show frames and stats
python3 unicorn_probe.py <MAC> --duration 600 --csv run.csv  # 10-minute integrity run with CSV
python3 unicorn_probe.py <MAC> --signal                    # per-channel RMS and alpha power every second
python3 unicorn_probe.py <MAC> --find-channel              # probe RFCOMM channels 1–30
```

On exit the tool prints a summary: rate, counter gaps, loss %, and a histogram of time between reads.

### 2. Flash the classic ESP32

```bash
cd firmware/bridge_esp32
pio run -t menuconfig        # "Unicorn bridge": Unicorn MAC, UART pins, GPIOs, options
pio run -t upload
pio device monitor
```

Default wiring and settings:

| Signal | Pin |
|---|---|
| UART TX → S3 RX (921600 8N1) | GPIO17 |
| Status out (high = streaming) | GPIO4 |
| Status LED | GPIO2 |
| Stop / resume button | GPIO0 (BOOT) |

LED patterns: off = disconnected, 1 Hz = connecting, 5 Hz = waiting for ACK, solid = streaming.

Every 5 s the log prints connection and validation stats, including the inquiry RSSI. If the bridge cannot connect, check that line first and move the ESP32 closer to the headset.

## Unicorn protocol summary

Bluetooth SPP, start `61 7C 87`, stop `63 5C C5` (both answered with `00 00 00`). Each sample is one 45-byte frame:
- header `C0 00`;
- battery;
- 8 × 24-bit big-endian EEG channels;
- 3-axis accelerometer and 3-axis gyroscope (int16 little-endian);
- uint32 little-endian sample counter;
- terminator `0D 0A`.

Channels: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. See `firmware/common/unicorn_protocol.h` for offsets and scaling.

References: Robert Oostenveld's `unicorn2lsl`, the Rust crate `gtec`, and g.tec's `UnicornBluetoothProtocol.pdf` ([unicorn-bi/Unicorn-Suite-Hybrid-Black](https://github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black)).
