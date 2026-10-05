# INTUNE

**I**ntegrated **N**eural **T**elemetry for **U**nsafe-state **N**otification and **E**mergency response

*Keeping robots in tune with the people who operate them.*

BR41N.IO Hackathon project (IEEE SMC 2026, Programming category: "Your Hacking Project").

A fully embedded pipeline (no PC, no g.tec dongle) that monitors an operator's EEG while they work with a robotic arm. It detects **error-related potentials (ErrPs)**, the brain response that appears when the operator sees the arm do something wrong, and sends **graded alerts** to the arm so it can react, for example slowing down on a mild alert or stopping on a severe one. The target is industrial use, but the design is general purpose: any setting where a person operates or supervises a robotic arm.

> Hackathon prototype. Not a certified safety system.

## Architecture

As built for the demo, the ErrP detector runs on the arm's supervisor ESP32; no ESP32-S3 is needed.

```
Unicorn Hybrid Black ──BT Classic SPP──> bridge ESP32 ───UART 921600 (EEG + EVENT)──> supervisor ESP32 ──UART──> RoArm-M2-S
   8 EEG ch @ 250 Hz                     validates frames,          ^                 ErrP-AE (TFLite Micro    (factory firmware)
                                         causal IIR 1–15 Hz,        |                 int8), artifact gate,
                                         gaps flagged (ZOH),        └──GPIO sync──────  graded alerts, safety
                                         sync pulse -> sample #        pulse at t = 0    (slow / pause / stop)
```

| Node | Role |
|---|---|
| **Bridge (classic ESP32)** | Connects to the g.tec Unicorn Hybrid Black over Bluetooth Classic. Parses and validates frames, fills short gaps by zero-order hold and flags them, filters each channel with a causal **1–15 Hz IIR band-pass**, and streams `link_protocol.h` frames over UART. Converts the arm's sync pulse (GPIO18) into a Unicorn sample index (`EVENT` frame). |
| **Supervisor (classic ESP32)** | Drives the RoArm-M2-S over UART in short steps. Fires the sync pulse at the onset of every action. With `CONFIG_ARM_LOCAL_DETECTOR` it also receives the EEG, cuts a [−200, +800] ms epoch per action, applies the artifact gate, runs the **ErrP autoencoder** (C2's engine, int8 TFLite Micro, ~49 k parameters, 8.4 KB arena) and turns the score into an alert level. |
| **ESP32-S3** (optional) | Original plan for the detector. Its firmware currently only runs the model's on-board tests; the same engine and weights are shared with the supervisor. |

### Why ErrPs

When a person sees a machine make a mistake, the EEG shows an error-related potential: a fronto-central negativity about 200–300 ms after the event, followed by a positivity. The Unicorn's Fz and Cz electrodes sit right where it is strongest. Detecting it lets the arm react to the operator noticing a problem.

### Alert levels

| Level | Criterion (per action) | Arm reaction |
|---|---|---|
| 0 | clean epoch, score ≤ T1 | Nominal speed |
| 1 | T1 < score ≤ T2 | Slows down |
| 2 | T2 < score ≤ T3, or 3 rejected epochs in a row | Pause until the operator confirms (BOOT short press) |
| 3 | score > T3, 2 epochs in a row > T2, or EEG lost | Safe stop until the operator re-arms (BOOT long press) |

The system is fail-safe: no EEG means no heartbeat, and the arm holds its position. Losing the EEG for more than 1 s after streaming started triggers level 3. Confirming or re-arming resets the detector to level 0, but only while the EEG is present.

## Status

| Step | Status |
|---|---|
| PC ↔ Unicorn over RFCOMM (protocol verified) | ✅ 0 % loss at 250 Hz |
| ESP32 ↔ Unicorn: discovery, connection, start/ACK | ✅ |
| 10-minute integrity test on the ESP32 | ✅ 0.001 % loss (2 of 176,197 samples), 0 corrupt frames, 0 reconnections |
| Signal check (alpha with eyes closed, jaw artifact) | ✅ Through the bridge and its IIR (`link_view.py live`) |
| IIR filter on the ESP32 vs `scipy.signal.sosfilt` | ✅ max error 0.00009 µV after settling (criterion < 0.01 µV) |
| Sync pulse → Unicorn sample index | ✅ < 2 ms interval error against the supervisor's clock |
| UART bridge → supervisor | ✅ 0 CRC errors |
| Recording paradigm with the real arm (v3.1) | ✅ 490 actions in one session, 99.6 % clean epochs |
| ErrP autoencoder on the supervisor, end to end | ✅ runs live; arm slows, pauses and stops on its levels |
| ErrP detection quality | ⚠️ **AUC ≈ 0.6** (0.57 autoencoder, 0.59 LDA baseline), single session |

### Results and findings

- **Model:** `ml/autoencoder/models/S04_v31/`, trained on session S04 (238 correct, 87 error actions). AUC ≈ 0.6: it separates errors from correct actions only slightly better than chance. In a 12-action live demo it stopped the arm on 1 of 2 scorable errors, and also raised 1 false stop and 2 false alerts. Treat it as a proof of the architecture, not a validated error detector. The project's go/no-go target was AUC > 0.6 with a significant ErrP; it was not reached.
- **No clear ErrP so far.** Over 4 recording sessions (~1,800 actions), the error − correct grand average on Fz/Cz never reached significance. Realistic single-trial ErrP accuracy with dry electrodes is low, and more data and better contact are needed.
- **Paradigm confound found and fixed (v3.1).** In session S03 the classes already differed at 0–150 ms, which an ErrP cannot do. Two causes: return movements after an error were labelled "correct", and the rest before each action depended on the previous movement. v3.1 labels those returns `recovery` (excluded) and measures the rest from the moment the arm stops. In S04 the classes match at 0–150 ms (p = 0.36) and the rest is identical (2.00 vs 2.03 s).
- **Electrode contact is the main noise source.** Cz (and C3/C4 next to it) lost contact mid-session twice, jumping from ~9 to ~30 µV. The recorder now runs a signal check (20 actions, robust noise ≤ 20 µV on Fz/Cz/C3/C4/Pz) before every block of 100.
- **No pairing needed.** The Unicorn accepts an RFCOMM connection on channel 1 with no PIN or SSP.
- **Radio range matters.** The PCB antenna of a WROOM-32 drops the headset easily. Keep the bridge right next to the operator, or use a WROOM-32U with an external antenna. If it stops connecting, power-cycle the headset.

## Repository layout

```
tools/linux_probe/                   # PC tools: unicorn_probe.py (headset direct), link_view.py (bridge stream), design_iir.py
tools/recording/                     # record_errp.py, run_session_v3.sh, live_view.py, demo_detector.py
firmware/common/                     # Shared headers: Unicorn protocol + parser, link_protocol.h, eeg_iir.h, arm_protocol.h
firmware/bridge_esp32/               # Bridge: BT acquisition, IIR filter, gap handling, link frames, sync input
firmware/arm_esp32/                  # Supervisor: arm control, safety, sync pulse, local ErrP detector (detector.c)
firmware/detector_s3/                # ESP32-S3 project + the shared autoencoder engine and exported weights
ml/data/                             # build_dataset.py, FORMAT.md (dataset contract)
ml/autoencoder/                      # ErrP-AE training (TensorFlow) and saved models
ml/c_exporter/                       # Keras model -> int8 C header
ml/eval/                             # Per-session reports (grand average, AUC)
```

## Getting started

### Requirements

- Linux with BlueZ (tested with Python 3.14, standard library only).
- [PlatformIO](https://platformio.org/). We install it with `uv tool install --python 3.12 platformio`.
- g.tec Unicorn Hybrid Black, two classic ESP32 boards (bridge and supervisor, e.g. ESP32-DevKitC / WROOM-32) and a Waveshare RoArm-M2-S.
- For recording and training: a Python 3.12 venv in the repo root:
  `uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install tensorflow numpy scipy pandas matplotlib scikit-learn pyserial`.

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
| UART TX → supervisor RX GPIO26 (921600 8N1) | GPIO17 |
| Sync pulse in (from supervisor GPIO25, rising edge) | GPIO18 |
| Status out (high = streaming) | GPIO4 |
| Status LED | GPIO2 |
| Stop / resume button | GPIO0 (BOOT) |

LED patterns: off = disconnected, 1 Hz = connecting, 5 Hz = waiting for ACK, solid = streaming.

By default the bridge also mirrors its link frames on USB (`BRIDGE_LINK_MIRROR_CONSOLE`), so `pio device monitor` shows no logs; read the stream with `tools/linux_probe/link_view.py live`. Disable that option in menuconfig to see the text logs again.

### 3. Flash the supervisor

```bash
cd firmware/arm_esp32
pio run -t upload --upload-port /dev/serial/by-id/<supervisor>
```

Wiring: RoArm pin 10 → GPIO16, pin 8 → GPIO17, pin 6 → GND; bridge GPIO17 → GPIO26 (EEG); GPIO25 → bridge GPIO18 (sync pulse); common GND. Console at 115200: `status`, `det` (detector stats), `home`, `act`, `s3 <n>` (simulate the S3; silences the detector).

### 4. Record a dataset session and train

```bash
BRIDGE=/dev/serial/by-id/<bridge> ARM=/dev/serial/by-id/<supervisor> bash tools/recording/run_session_v3.sh
```

10 familiarization actions, then 6 × (20-action signal check + 100 dataset actions + rest). A live view opens on its own. At the end the script builds the dataset, writes the ErrP report to `ml/eval/` and trains a new model in `ml/autoencoder/models/<SUBJECT>_v31/`. Export it to C with `ml/c_exporter/export_to_c.py`.

### 5. Live demo

```bash
.venv/bin/python tools/recording/demo_detector.py --port /dev/serial/by-id/<supervisor>
```

The arm loops left → forward → right → back and makes deliberate random errors 25 % of the time. Each action prints the supervisor's score and level. After a pause or stop the arm waits 5 s and resumes on its own (`--resume-after`, `--manual` to require the BOOT button).

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
