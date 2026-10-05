"""Genera los headers C del Detector (S3) a partir de los artefactos de ML.

    uv run --project ml/autoencoder ml/autoencoder/training/export_headers.py

Entradas (corre antes train.py, export_tflite.py, evaluate.py y baseline_lda.py):
    models/errp_ae_int8.tflite, models/errp_ae_meta.json, config.json, models/lda.json
Salidas en firmware/detector_s3/:
    include/errp_model_data.h, src/errp_model_data.c   modelo int8 para TFLite Micro
    include/errp_params.h                               preprocesamiento, normalización
                                                        y umbrales por defecto, LDA
    include/errp_golden.h                               epochs de referencia (prueba 12)
Los valores de normalización y umbrales son los offline: el S3 los reemplaza
con su calibración por sesión (NVS).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

AE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE_DIR))
import errp_pipeline as ep  # noqa: E402
from baseline_lda import features as lda_features  # noqa: E402

REPO = AE_DIR.parents[1]
FW = REPO / "firmware" / "detector_s3"
TFLITE = AE_DIR / "models" / "errp_ae_int8.tflite"
META = AE_DIR / "models" / "errp_ae_meta.json"
CONFIG = AE_DIR / "config.json"
LDA = AE_DIR / "models" / "lda.json"

BANNER = "// GENERADO por ml/autoencoder/training/export_headers.py. No editar a mano.\n"


def f32(v) -> str:
    s = f"{float(np.float32(v)):.9g}"
    if not any(c in s for c in ".eEn"):  # "100f" no es un literal C válido
        s += ".0"
    return s + "f"


def farr(values, per_line=8, indent="    ") -> str:
    vals = [f32(v) for v in np.asarray(values, dtype=np.float32).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return "{\n" + ",\n".join(indent + ln for ln in lines) + "\n}"


def matrix(rows, per_line=8) -> str:
    """Arreglo 2D con llaves anidadas por fila (-Werror=missing-braces en ESP-IDF)."""
    return "{\n" + ",\n".join("    " + farr(r, per_line, "        ").replace("\n}", "\n    }") for r in rows) + "\n}"


def main():
    for p in (TFLITE, META, CONFIG, LDA):
        if not p.exists():
            raise SystemExit(f"falta {p}: corre el pipeline completo antes")
    model = TFLITE.read_bytes()
    meta = json.loads(META.read_text())
    cfg = json.loads(CONFIG.read_text())
    lda = json.loads(LDA.read_text())
    tfl = meta["tflite"]
    if not tfl["corr_pass"]:
        raise SystemExit(f"el int8 no pasó la correlación ({tfl['float_int8_corr']:.4f}); no se exporta")
    synthetic = int(bool(cfg.get("synthetic_data")))
    src_note = f"// Datos de entrenamiento: {Path(meta['data']).name}" + (" (SINTÉTICOS)" if synthetic else "") + "\n"

    (FW / "include").mkdir(parents=True, exist_ok=True)
    (FW / "src").mkdir(parents=True, exist_ok=True)

    # --- modelo ---
    (FW / "include" / "errp_model_data.h").write_text(
        BANNER + src_note + f"""#pragma once
#include <stddef.h>
#include <stdint.h>

#define ERRP_MODEL_ARCH        "{meta['arch']}"
#define ERRP_MODEL_PARAMS      {meta['params']}
#define ERRP_MODEL_SYNTHETIC   {synthetic}
// Ops: {', '.join(tfl['ops'])}

// Cuantización (entrada y salida int8, forma [1, 8, 40, 1] NHWC)
#define ERRP_IN_SCALE          {f32(tfl['input']['scale'])}
#define ERRP_IN_ZERO_POINT     {tfl['input']['zero_point']}
#define ERRP_OUT_SCALE         {f32(tfl['output']['scale'])}
#define ERRP_OUT_ZERO_POINT    {tfl['output']['zero_point']}
#define ERRP_FLOAT_INT8_CORR   {f32(tfl['float_int8_corr'])}

extern const uint8_t g_errp_model[];
extern const size_t g_errp_model_len;
""")
    hexrows = [", ".join(f"0x{b:02x}" for b in model[i:i + 16]) for i in range(0, len(model), 16)]
    (FW / "src" / "errp_model_data.c").write_text(
        BANNER + src_note + '#include "errp_model_data.h"\n\n'
        "// TFLite Micro exige el flatbuffer alineado a 16 B\n"
        f"alignas(16) const uint8_t g_errp_model[] = {{\n    " + ",\n    ".join(hexrows) + "\n};\n"
        f"const size_t g_errp_model_len = {len(model)};\n")

    # --- parámetros ---
    sos = signal.butter(2, [1, 15], btype="band", fs=ep.FS, output="sos")
    zi = signal.sosfilt_zi(sos)
    thr = cfg["thresholds"]
    pre = cfg["preprocessing"]
    edges = lda["features"]["bin_edges_samples_250hz"]
    (FW / "include" / "errp_params.h").write_text(
        BANNER + src_note + f"""#pragma once
#include <stdint.h>

// Preprocesamiento (idéntico a ml/autoencoder/errp_pipeline.py)
// Filtro (en C1): {pre['filter']}
#define ERRP_FS_HZ         {ep.FS}
#define ERRP_N_CH          {ep.N_CH}
#define ERRP_PRE           {ep.PRE}     // muestras de baseline, [-200, 0) ms
#define ERRP_POST          {ep.POST}    // muestras de [0, 800) ms
#define ERRP_DECIM         {ep.DECIM}     // media de cada 5 -> 50 Hz
#define ERRP_N_T           {ep.N_T}    // muestras por canal que ve el modelo
#define ERRP_GATE_UV       {f32(ep.GATE_UV)}
#define ERRP_FLAT_STD_UV   {f32(ep.FLAT_STD_UV)}

// Normalización z-score por canal (offline; la calibración del S3 la reemplaza)
static const float ERRP_DEFAULT_MEAN[ERRP_N_CH] = {farr(cfg['normalization']['mean'])};
static const float ERRP_DEFAULT_STD[ERRP_N_CH] = {farr(cfg['normalization']['std'])};

// Umbrales del score MSE (offline; la calibración del S3 los reemplaza)
#define ERRP_DEFAULT_T1    {f32(thr['T1'])}  // p{thr['percentiles']['T1']:g}
#define ERRP_DEFAULT_T2    {f32(thr['T2'])}  // p{thr['percentiles']['T2']:g}
#define ERRP_DEFAULT_T3    {f32(thr['T3'])}  // p{thr['percentiles']['T3']:g}
#define ERRP_PCT_T1        {f32(thr['percentiles']['T1'])}
#define ERRP_PCT_T2        {f32(thr['percentiles']['T2'])}
#define ERRP_PCT_T3        {f32(thr['percentiles']['T3'])}

// Baseline LDA (scorer de respaldo): score = w . f + b
// f[ch * {lda['features']['n_bins']} + bin] = media µV (baseline restado) en [edges[bin], edges[bin+1]) muestras desde t = 0
#define ERRP_LDA_N_BINS    {lda['features']['n_bins']}
static const int16_t ERRP_LDA_EDGES[ERRP_LDA_N_BINS + 1] = {{ {', '.join(map(str, edges))} }};
static const float ERRP_LDA_W[ERRP_N_CH * ERRP_LDA_N_BINS] = {farr(lda['w'])};
#define ERRP_LDA_B         {f32(lda['b'])}
#define ERRP_LDA_AUC       {f32(lda['evaluation']['auc'])}

// IIR causal 1-15 Hz (regla dura 1). Lo aplica C1; el S3 lo usa SOLO en el modo
// provisional de tramas Unicorn crudas (DETECTOR_INPUT_RAW_UNICORN).
// Por sección: b0 b1 b2 a0 a1 a2 (a0 = 1). Estado inicial = ERRP_IIR_ZI * x0 (sosfilt_zi).
#define ERRP_IIR_N_SOS     {len(sos)}
static const float ERRP_IIR_SOS[ERRP_IIR_N_SOS][6] = {matrix(sos, 6)};
static const float ERRP_IIR_ZI[ERRP_IIR_N_SOS][2] = {matrix(zi, 2)};
""")

    # --- epochs de referencia ---
    gold = cfg["golden"]
    # DSP: un epoch crudo de test aceptado por el gate -> preprocesado y LDA esperados
    d = ep.load_epochs(Path(meta["data"]))
    _, _, te = ep.chrono_split(len(d["X"]))
    i_raw = te[ep.gate(d)[te]][0]
    raw = d["X"][i_raw:i_raw + 1]
    pre_ref = ep.preprocess(raw)[0]
    lda_ref = float(lda_features(raw)[0] @ np.float32(lda["w"]) + np.float32(lda["b"]))
    # IIR: señal con offset DC grande, como el Unicorn crudo
    rng = np.random.default_rng(7)
    x_iir = (15000.0 + np.cumsum(rng.normal(0, 3, 500)) + 20 * np.sin(2 * np.pi * 6 * np.arange(500) / ep.FS))
    y_iir, _ = signal.sosfilt(sos, x_iir, zi=zi * x_iir[0])
    (FW / "include" / "errp_golden.h").write_text(
        BANNER + src_note + f"""#pragma once
// Epochs ya normalizados [8][40] y su score esperado (prueba 12: online == offline).
#define ERRP_N_GOLDEN {len(gold)}
static const float ERRP_GOLDEN_INPUT[ERRP_N_GOLDEN][8][40] = {{
{','.join(chr(10) + '  ' + matrix(g['input_norm']).replace(chr(10), chr(10) + '  ') for g in gold)}
}};
static const float ERRP_GOLDEN_SCORE_INT8[ERRP_N_GOLDEN] = {{ {', '.join(f32(g['score_int8']) for g in gold)} }};
static const float ERRP_GOLDEN_SCORE_FLOAT[ERRP_N_GOLDEN] = {{ {', '.join(f32(g['score_float']) for g in gold)} }};
static const int ERRP_GOLDEN_LABEL[ERRP_N_GOLDEN] = {{ {', '.join(str(g['label']) for g in gold)} }};

// DSP: epoch crudo [8][260] (µV filtrados, t = 0 en el índice 55) -> preprocesado
// [8][40] sin normalizar y score LDA esperados.
#define ERRP_GOLDEN_RAW_LEN {ep.STORED_LEN}
#define ERRP_GOLDEN_RAW_T0  {ep.T0_INDEX}
static const float ERRP_GOLDEN_RAW[8][ERRP_GOLDEN_RAW_LEN] = {matrix(raw[0])};
static const float ERRP_GOLDEN_RAW_PRE[8][40] = {matrix(pre_ref)};
#define ERRP_GOLDEN_RAW_LDA {f32(lda_ref)}

// IIR: entrada (DC grande + deriva + 6 Hz) y salida de scipy sosfilt con zi * x[0]
#define ERRP_GOLDEN_IIR_LEN {len(x_iir)}
static const float ERRP_GOLDEN_IIR_X[ERRP_GOLDEN_IIR_LEN] = {farr(x_iir)};
static const float ERRP_GOLDEN_IIR_Y[ERRP_GOLDEN_IIR_LEN] = {farr(y_iir)};
""")
    print(f"modelo {len(model)} B ({meta['arch']}), sintético={bool(synthetic)} -> {FW.relative_to(REPO)}/"
          "{include/errp_model_data.h, src/errp_model_data.c, include/errp_params.h, include/errp_golden.h}")


if __name__ == "__main__":
    main()
