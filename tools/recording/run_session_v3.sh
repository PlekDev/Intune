#!/usr/bin/env bash
# INTUNE v3: una sola grabación con familiarización y pruebas de señal por bloque, dataset nuevo y
# modelo nuevo. Mismo paradigma v2 (error = postura aleatoria a >= 45° del destino correcto).
#   1. 10 acciones sin errores para memorizar el recorrido (label familiar) + pausa de 30 s
#   2. 6 bloques: prueba de señal de 20 acciones sin errores (label check) -> 100 acciones del dataset
#      Si la prueba sale mal (ruido robusto > 20 µV en Fz, Cz, C3, C4 o Pz, pulsos o pérdidas), pide
#      acomodar la diadema y la repite (hasta 3 veces). Resultados en checks.csv.
#   3. dataset NUEVO solo con esta sesión       -> ml/data/processed/<MODELO>/
#   4. reporte del ErrP                         -> ml/eval/report_<sesión>.md (+ .png)
#   5. modelo NUEVO                             -> ml/autoencoder/models/<MODELO>/
# familiar y check quedan en events.csv pero nunca entran al dataset (y = -1).
#
# Uso (Linux, desde la raíz del repo, en una terminal: pide Enter si una prueba sale mal):
#   BRIDGE=/dev/serial/by-id/<puente> ARM=/dev/serial/by-id/<supervisor> bash tools/recording/run_session_v3.sh
# Variables opcionales: SUJ (def. S03), N (def. 600), MODELO (def. <SUJ>_v3), PY (def. .venv/bin/python),
#   CHECK_UV (def. 20)
# Cableado: supervisor GPIO25 -> puente GPIO18, GND con GND. Diadema puesta y encendida.
# Ctrl+C corta y guarda lo grabado (los pasos 3-5 se siguen corriendo con lo que haya).
set -euo pipefail
cd "$(dirname "$0")/../.."

SUJ=${SUJ:-S03}
N=${N:-600}
MODELO=${MODELO:-${SUJ}_v3}
PY=${PY:-.venv/bin/python}
CHECK_UV=${CHECK_UV:-20}
: "${BRIDGE:?falta BRIDGE=/dev/serial/by-id/... (puente). Mira: ls -l /dev/serial/by-id/}"
: "${ARM:?falta ARM=/dev/serial/by-id/... (supervisor del brazo)}"

if [ -e "ml/autoencoder/models/$MODELO" ]; then
    echo "ya existe ml/autoencoder/models/$MODELO: usa otro MODELO=..." >&2
    exit 1
fi

echo "=== 1-2/5: grabación v3 ($SUJ): 10 familiarización + 6 x (20 prueba + 100 dataset)"
"$PY" tools/recording/record_errp.py --no-blocks --bridge-port "$BRIDGE" --arm-port "$ARM" \
    --subject "$SUJ" --n-actions "$N" --p-error 0.22 \
    --familiar 10 --familiar-pause 30 \
    --check-every 100 --check-n 20 --check-max-uv "$CHECK_UV" \
    --notes "sesion v3: familiarizacion 10 + prueba de senal de 20 antes de cada bloque de 100"

SES=$(ls -td ml/data/raw/*_"$SUJ" | head -1)
echo "sesión: $SES"
if [ -f "$SES/checks.csv" ]; then
    echo "pruebas de señal (block, try, passed, ...):"
    cut -d, -f1-3,6-9 "$SES/checks.csv"
fi

echo "=== 3/5: dataset nuevo"
"$PY" ml/data/build_dataset.py "$SES" --out "ml/data/processed/$MODELO" \
    --export-c2 "ml/data/processed/$MODELO/c2_train.npz"

echo "=== 4/5: reporte del ErrP"
"$PY" ml/eval/first_report.py "ml/data/processed/$MODELO/$(basename "$SES").npz"

echo "=== 5/5: modelo nuevo"
"$PY" ml/autoencoder/train_errp_ae.py --data "ml/data/processed/$MODELO/c2_train.npz" \
    --out "ml/autoencoder/models/$MODELO"

echo "=== listo: modelo en ml/autoencoder/models/$MODELO, reporte en ml/eval/report_$(basename "$SES").md"
echo "Para subirlo: git add ml/autoencoder/models/$MODELO ml/eval/*$(basename "$SES")* && git commit -m \"modelo $MODELO\" && git push"
