"""Genera los headers C del Detector (S3) a partir de los artefactos de ML.

    uv run --project ml/autoencoder ml/autoencoder/training/export_headers.py

Entradas (corre antes baseline_lda.py, train.py, export_tflite.py y evaluate.py):
    models/errp_ae_int8.tflite, models/errp_ae_meta.json, config.json, models/lda.json,
    y una sesión .npz de C4 (gate_uv, flat_std_uv, gate_gyro con los que se construyó).
Salidas en firmware/detector_s3/:
    include/errp_model_data.h, src/errp_model_data.c   modelo int8 para TFLite Micro
    include/errp_params.h                               gate, normalización y umbrales por
                                                        defecto, LDA, coeficientes IIR
    include/errp_golden.h                               vectores de referencia (prueba 12 y DSP)
La normalización y los umbrales son los offline: el S3 los reemplaza con su
calibración por sesión (NVS).
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


def nested(blocks) -> str:
    return "{" + ",".join("\n  " + matrix(b).replace("\n", "\n  ") for b in blocks) + "\n}"


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
    d0 = np.load(meta["data"][0], allow_pickle=False)
    gate_uv, flat_uv, gate_gyro = float(d0["gate_uv"]), float(d0["flat_std_uv"]), float(d0["gate_gyro"])
    synthetic = int(bool(cfg.get("synthetic_data")))
    names = ", ".join(Path(p).stem for p in meta["data"])
    src_note = f"// Datos: {names}" + (" (SINTÉTICOS)" if synthetic else "") + f"; dataset {d0['pipeline_version']}\n"

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
        "alignas(16) const uint8_t g_errp_model[] = {\n    " + ",\n    ".join(hexrows) + "\n};\n"
        f"const size_t g_errp_model_len = {len(model)};\n")

    # --- parámetros ---
    sos = signal.butter(2, [1, 15], btype="band", fs=ep.FS, output="sos")
    zi = signal.sosfilt_zi(sos)
    thr = cfg["thresholds"]
    edges = lda["features"]["bin_edges_samples_50hz"]
    (FW / "include" / "errp_params.h").write_text(
        BANNER + src_note + f"""#pragma once
#include <stdint.h>

// Cadena de FORMAT.md §2 (C4) / CLAUDE.md "Signal chain"
#define ERRP_FS_HZ         {ep.FS}
#define ERRP_N_CH          {ep.N_CH}
#define ERRP_PRE           {ep.PRE}     // muestras de baseline, [-200, 0) ms
#define ERRP_POST          {ep.POST}    // muestras de [0, 800) ms
#define ERRP_DECIM         {ep.DECIM}     // media de cada 5 -> 50 Hz
#define ERRP_N_T           {ep.N_T}    // muestras por canal de X
#define ERRP_STD_FLOOR     {f32(ep.STD_FLOOR)}

// Puerta de artefactos (FORMAT.md §3), los valores con los que C4 construyó el dataset
#define ERRP_GATE_UV       {f32(gate_uv)}   // max |X| tras baseline y diezmado
#define ERRP_FLAT_STD_UV   {f32(flat_uv)}   // std de un canal de X
#define ERRP_GATE_GYRO_DPS {f32(gate_gyro)}   // rango pico a pico por eje (máx. de 3), ventana de 250

// Normalización z-score por canal (promedio offline; la calibración del S3 la reemplaza)
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
// f[ch * {lda['features']['n_bins']} + bin] = media de X[ch][edges[bin] .. edges[bin+1]) (µV, 50 Hz)
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

    # --- vectores de referencia ---
    gold = cfg["golden"]
    rng = np.random.default_rng(7)
    # Ventana [8, 250] µV filtrados: ruido + ErrP de juguete en Fz/Cz + offset distinto por canal
    t = np.arange(ep.PRE + ep.POST) / ep.FS - 0.2
    win = (rng.normal(0, 6, (ep.N_CH, len(t))) + rng.uniform(-20, 20, (ep.N_CH, 1))
           - 8 * np.exp(-0.5 * ((t - 0.25) / 0.04) ** 2) * np.array([1, .4, 1, .4, .3, 0, 0, 0])[:, None])
    x_ref = ep.preprocess_window(win)
    lda_ref = float(lda_features(x_ref[None])[0] @ np.float32(lda["w"]) + np.float32(lda["b"]))
    x_iir = (15000.0 + np.cumsum(rng.normal(0, 3, 500)) + 20 * np.sin(2 * np.pi * 6 * np.arange(500) / ep.FS))
    y_iir, _ = signal.sosfilt(sos, x_iir, zi=zi * x_iir[0])
    have_int8 = all("score_int8" in g for g in gold)
    (FW / "include" / "errp_golden.h").write_text(
        BANNER + src_note + f"""#pragma once
// Épocas de test de C4: X [8][40] µV, normalización de su sesión, entrada normalizada
// y scores esperados (prueba 12: online == offline).
#define ERRP_N_GOLDEN {len(gold)}
static const float ERRP_GOLDEN_X[ERRP_N_GOLDEN][8][40] = {nested([g['x_uv'] for g in gold])};
static const float ERRP_GOLDEN_MEAN[ERRP_N_GOLDEN][8] = {matrix([g['mean'] for g in gold])};
static const float ERRP_GOLDEN_STD[ERRP_N_GOLDEN][8] = {matrix([g['std'] for g in gold])};
static const float ERRP_GOLDEN_INPUT[ERRP_N_GOLDEN][8][40] = {nested([g['input_norm'] for g in gold])};
static const float ERRP_GOLDEN_SCORE_INT8[ERRP_N_GOLDEN] = {{ {', '.join(f32(g['score_int8'] if have_int8 else 'nan') for g in gold)} }};
static const float ERRP_GOLDEN_SCORE_FLOAT[ERRP_N_GOLDEN] = {{ {', '.join(f32(g['score_float']) for g in gold)} }};
static const float ERRP_GOLDEN_LDA[ERRP_N_GOLDEN] = {{ {', '.join(f32(g['lda']) for g in gold)} }};
static const int ERRP_GOLDEN_LABEL[ERRP_N_GOLDEN] = {{ {', '.join(str(g['label']) for g in gold)} }};

// DSP: ventana [8][250] µV filtrados ([-200, +800) ms) -> X [8][40] y LDA esperados
static const float ERRP_GOLDEN_WIN[8][250] = {matrix(win)};
static const float ERRP_GOLDEN_WIN_X[8][40] = {matrix(x_ref)};
#define ERRP_GOLDEN_WIN_LDA {f32(lda_ref)}

// IIR: entrada (DC grande + deriva + 6 Hz) y salida de scipy sosfilt con zi * x[0]
#define ERRP_GOLDEN_IIR_LEN {len(x_iir)}
static const float ERRP_GOLDEN_IIR_X[ERRP_GOLDEN_IIR_LEN] = {farr(x_iir)};
static const float ERRP_GOLDEN_IIR_Y[ERRP_GOLDEN_IIR_LEN] = {farr(y_iir)};
""")
    print(f"modelo {len(model)} B ({meta['arch']}), sintético={bool(synthetic)}, gate {gate_uv:g} µV / "
          f"flat {flat_uv:g} µV / gyro {gate_gyro:g} °/s -> {FW.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
