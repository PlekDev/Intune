#!/usr/bin/env bash
# INTUNE: sesión completa con el paradigma v2 y modelo nuevo, desde cero.
#   1. 10 acciones normales (sin errores)      -> ml/data/raw/<fecha>_<SUJ>_normal/
#   2. pausa de 30 s
#   3. N acciones con ~22 % de errores (v2)    -> ml/data/raw/<fecha>_<SUJ>/
#   4. dataset NUEVO solo con esa sesión       -> ml/data/processed/<MODELO>/
#   5. reporte del ErrP                        -> ml/eval/report_<sesión>.md (+ .png)
#   6. modelo NUEVO                            -> ml/autoencoder/models/<MODELO>/
#
# Uso (Linux, desde la raíz del repo):
#   BRIDGE=/dev/serial/by-id/<puente> ARM=/dev/serial/by-id/<supervisor> bash tools/recording/run_session_v2.sh
# Variables opcionales: SUJ (def. S02), N (def. 600), MODELO (def. <SUJ>_v2), PY (def. .venv/bin/python)
# Cableado: supervisor GPIO25 -> puente GPIO18, GND con GND. Diadema puesta y encendida.
# Durante la sesión: Enter en cada descanso (cada 100 acciones). Ctrl+C corta y guarda lo grabado.
set -euo pipefail
cd "$(dirname "$0")/../.."

SUJ=${SUJ:-S02}
N=${N:-600}
MODELO=${MODELO:-${SUJ}_v2}
PY=${PY:-.venv/bin/python}
: "${BRIDGE:?falta BRIDGE=/dev/serial/by-id/... (puente). Mira: ls -l /dev/serial/by-id/}"
: "${ARM:?falta ARM=/dev/serial/by-id/... (supervisor del brazo)}"

if [ -e "ml/autoencoder/models/$MODELO" ]; then
    echo "ya existe ml/autoencoder/models/$MODELO: usa otro MODELO=..." >&2
    exit 1
fi

REC=(tools/recording/record_errp.py --no-blocks --bridge-port "$BRIDGE" --arm-port "$ARM")

echo "=== 1/6: 10 acciones normales ($SUJ)"
"$PY" "${REC[@]}" --subject "${SUJ}_normal" --n-actions 10 --p-error 0 \
    --notes "10 normales previas (paradigma v2)"

echo "=== 2/6: pausa de 30 s"
sleep 30

echo "=== 3/6: $N acciones con errores ($SUJ, paradigma v2)"
"$PY" "${REC[@]}" --subject "$SUJ" --n-actions "$N" --p-error 0.22 --rest-every 100 \
    --notes "sesion v2: error = postura aleatoria >= 45 grados; descansos cada 100"

SES=$(ls -td ml/data/raw/*_"$SUJ" | head -1)
echo "sesión: $SES"

echo "=== 4/6: dataset nuevo"
"$PY" ml/data/build_dataset.py "$SES" --out "ml/data/processed/$MODELO" \
    --export-c2 "ml/data/processed/$MODELO/c2_train.npz"

echo "=== 5/6: reporte del ErrP"
"$PY" ml/eval/first_report.py "ml/data/processed/$MODELO/$(basename "$SES").npz"

echo "=== 6/6: modelo nuevo"
"$PY" ml/autoencoder/train_errp_ae.py --data "ml/data/processed/$MODELO/c2_train.npz" \
    --out "ml/autoencoder/models/$MODELO"

echo "=== listo: modelo en ml/autoencoder/models/$MODELO, reporte en ml/eval/report_$(basename "$SES").md"
