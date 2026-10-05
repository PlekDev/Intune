# INTUNE C3: brazo (RoArm-M2-S) + supervisor

El brazo (Waveshare RoArm-M2-S) conserva su firmware de fábrica; **no se toca su ESP32**.
Lo controla una ESP32 aparte, el **supervisor** (`src/`), por UART. El supervisor hace los
movimientos por pasos cortos (si algo falla, el brazo se detiene en menos de 1°), aplica los
niveles de alerta del S3 y da el pulso de sync al S3 al inicio de cada acción.

## Mover el brazo de A a B desde tu laptop

Solo necesitas Python 3 y `pyserial`, y conectar **el supervisor** por USB a tu laptop
(nunca el USB-C del brazo).

```bash
python -m pip install -r firmware/arm_esp32/tools/requirements.txt
python firmware/arm_esp32/tools/move_ab.py
```

Se mueve de A a B y de vuelta **sin fin**. **Ctrl+C lo detiene en el acto** y el brazo queda
quieto con torque (no cae). En Windows usa `python` o `py`; en Linux/macOS puede ser `python3`.

Opciones (ángulos en grados: base, hombro, codo, pinza; postura de arranque `0,0,90,180`):

```bash
python firmware/arm_esp32/tools/move_ab.py --a -45,0,90,180 --b 45,20,110,180 --vel 25 --pausa 2
python firmware/arm_esp32/tools/move_ab.py --reps 3          # solo 3 idas y vueltas
python firmware/arm_esp32/tools/move_ab.py --sin-pulso       # sin pulso de sync
python firmware/arm_esp32/tools/move_ab.py --port COM5       # si no encuentra el supervisor solo
python firmware/arm_esp32/tools/move_ab.py --help
```

Límites: base ±91°, hombro −34..51°, codo 34..149°, pinza 92..183°. Velocidad máx. 57 °/s.

### Tres puntos (A → B → C → A …), movimiento más amplio

```bash
python firmware/arm_esp32/tools/move_abc.py
python firmware/arm_esp32/tools/move_abc.py --vel 30 --pausa 1.5
python firmware/arm_esp32/tools/move_abc.py --a -70,0,90,180 --b 0,25,115,130 --c 70,-10,80,180
```

Por defecto: A = `-60,0,90,180` (girado a un lado), B = `0,20,110,130` (centro, inclinado hacia
delante, pinza abierta), C = `60,-10,80,180` (girado al otro lado, levantado). Mismas opciones y
mismo Ctrl+C que `move_ab.py` (y necesita ese archivo al lado: comparten la lógica).

Si falla:
- *No encuentro el supervisor*: cable USB de datos, o indica el puerto con `--port`
  (Linux `/dev/ttyACM0` o `/dev/ttyUSB0`, macOS `/dev/cu.usbserial-…`/`/dev/cu.wchusbserial…`, Windows `COMx`).
  En Windows puede hacer falta el driver del chip USB (CH340/CH9102 o CP210x).
- *Linux, permiso denegado*: `sudo usermod -aG dialout $USER` y vuelve a iniciar sesión.
- *El brazo no responde por UART*: brazo encendido y terminado de arrancar (unos segundos), y cables bien.
- Cierra cualquier monitor serie que tenga abierto el puerto.

## Cableado

| Brazo (conector de 40 pines de su placa) | Supervisor (ESP32 DevKit) |
|---|---|
| pin 10 (TX del brazo) | RX2 / GPIO16 |
| pin 8 (RX del brazo) | TX2 / GPIO17 |
| pin 6 (GND) | GND |

| Supervisor | ESP32-S3 (C2) |
|---|---|
| GPIO25: pulso de sync, activo en alto 1 ms, bajo en reposo | pin de interrupción (flanco de subida, pull-down) |
| GPIO27 (TX) / GPIO26 (RX) | UART del S3 (`firmware/common/arm_protocol.h`, borrador) |
| GND | GND (masa común obligatoria) |

No conectes los pines de 5 V del conector del brazo (2 y 4), ni RX0/TX0 del supervisor (son su USB).

## Firmware del supervisor

```bash
cd firmware/arm_esp32 && pio run -t upload
```

PlatformIO con ESP-IDF 6.1. Los comandos de la consola del supervisor (115200 baudios) están listados en la cabecera de `src/main.c`.
`roarm_ref/` es el firmware de Waveshare, solo como referencia del protocolo JSON (no se graba).
