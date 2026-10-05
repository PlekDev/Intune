"""Genera los headers del S3 que NO son el modelo: gate, LDA, IIR y vectores del DSP.

    uv run --project ml/autoencoder ml/autoencoder/training/export_headers.py [--session <sesion>.npz]

El ErrP-AE (pesos, cuantización, normalización y umbrales por defecto, model_id) lo genera
ml/c_exporter/export_to_c.py en include/autoencoder_weights.h. Este script escribe:
    include/errp_params.h   puerta de artefactos (valores de la sesión de C4 con que se construyó
                            el dataset), LDA w[64]/b y bordes de bins, IIR 1-15 Hz del modo crudo
    include/errp_golden.h   vectores de referencia del DSP (preprocesado, LDA, IIR) para
                            test/host_dsp_test.c
Entradas: models/lda.json (baseline_lda.py) y una sesión .npz de C4 (gate_uv, flat_std_uv, gate_gyro).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

AE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE_DIR))
import errp_pipeline as ep  # noqa: E402
from baseline_lda import OUT_PATH as LDA  # noqa: E402
from baseline_lda import features as lda_features  # noqa: E402

REPO = AE_DIR.parents[1]
FW = REPO / "firmware" / "detector_s3"

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, help="sesión .npz de C4 (def. la primera de ml/data/processed)")
    ap.add_argument("--lda", type=Path, default=LDA)
    args = ap.parse_args()

    if not args.lda.exists():
        raise SystemExit(f"falta {args.lda}: corre baseline_lda.py antes")
    lda = json.loads(args.lda.read_text())
    session = args.session or ep.load_sessions(None)[0].path
    d0 = np.load(session, allow_pickle=False)
    gate_uv, flat_uv, gate_gyro = float(d0["gate_uv"]), float(d0["flat_std_uv"]), float(d0["gate_gyro"])
    src_note = f"// Gate de {Path(session).stem} ({d0['pipeline_version']}); LDA de {args.lda.name}\n"

    (FW / "include").mkdir(parents=True, exist_ok=True)
    sos = signal.butter(2, [1, 15], btype="band", fs=ep.FS, output="sos")
    zi = signal.sosfilt_zi(sos)
    edges = lda["features"]["bin_edges_samples_50hz"]
    (FW / "include" / "errp_params.h").write_text(
        BANNER + src_note + f"""#pragma once
#include <stdint.h>

// Cadena de FORMAT.md §2 (C4) / CLAUDE.md "Signal chain" (preprocesado: ae_preprocess)
#define ERRP_FS_HZ         {ep.FS}
#define ERRP_N_CH          {ep.N_CH}
#define ERRP_PRE           {ep.PRE}     // muestras de baseline, [-200, 0) ms
#define ERRP_POST          {ep.POST}    // muestras de [0, 800) ms
#define ERRP_N_T           {ep.N_T}    // muestras por canal de X (50 Hz)

// Puerta de artefactos (FORMAT.md §3), los valores con los que C4 construyó el dataset
#define ERRP_GATE_UV       {f32(gate_uv)}   // max |X| tras baseline y diezmado
#define ERRP_FLAT_STD_UV   {f32(flat_uv)}   // std de un canal de X
#define ERRP_GATE_GYRO_DPS {f32(gate_gyro)}   // rango pico a pico por eje (máx. de 3), ventana de 250

// Baseline LDA (scorer de respaldo): score = w . f + b
// f[ch * {lda['features']['n_bins']} + bin] = media de X[ch][edges[bin] .. edges[bin+1]) (µV, 50 Hz)
#define ERRP_LDA_N_BINS    {lda['features']['n_bins']}
static const int16_t ERRP_LDA_EDGES[ERRP_LDA_N_BINS + 1] = {{ {', '.join(map(str, edges))} }};
static const float ERRP_LDA_W[ERRP_N_CH * ERRP_LDA_N_BINS] = {farr(lda['w'])};
#define ERRP_LDA_B         {f32(lda['b'])}
#define ERRP_LDA_AUC       {f32(lda['evaluation']['auc'])}
#define ERRP_LDA_PCT_T1    90.0f
#define ERRP_LDA_PCT_T2    97.0f
#define ERRP_LDA_PCT_T3    99.0f

// IIR causal 1-15 Hz (regla dura 1). Lo aplica C1; el S3 lo usa SOLO en el modo
// provisional de tramas Unicorn crudas (DETECTOR_INPUT_RAW_UNICORN).
// Por sección: b0 b1 b2 a0 a1 a2 (a0 = 1). Estado inicial = ERRP_IIR_ZI * x0 (sosfilt_zi).
#define ERRP_IIR_N_SOS     {len(sos)}
static const float ERRP_IIR_SOS[ERRP_IIR_N_SOS][6] = {matrix(sos, 6)};
static const float ERRP_IIR_ZI[ERRP_IIR_N_SOS][2] = {matrix(zi, 2)};
""")

    rng = np.random.default_rng(7)
    # Ventana [8, 250] µV filtrados: ruido + ErrP de juguete en Fz/Cz + offset distinto por canal
    t = np.arange(ep.PRE + ep.POST) / ep.FS - 0.2
    win = (rng.normal(0, 6, (ep.N_CH, len(t))) + rng.uniform(-20, 20, (ep.N_CH, 1))
           - 8 * np.exp(-0.5 * ((t - 0.25) / 0.04) ** 2) * np.array([1, .4, 1, .4, .3, 0, 0, 0])[:, None])
    x_ref = ep.preprocess_window(win)
    lda_ref = float(lda_features(x_ref[None])[0] @ np.float32(lda["w"]) + np.float32(lda["b"]))
    x_iir = (15000.0 + np.cumsum(rng.normal(0, 3, 500)) + 20 * np.sin(2 * np.pi * 6 * np.arange(500) / ep.FS))
    y_iir, _ = signal.sosfilt(sos, x_iir, zi=zi * x_iir[0])
    (FW / "include" / "errp_golden.h").write_text(
        BANNER + src_note + f"""#pragma once
// DSP: ventana [8][250] µV filtrados ([-200, +800) ms) -> X [8][40] (ae_preprocess) y LDA esperados
static const float ERRP_GOLDEN_WIN[8][250] = {matrix(win)};
static const float ERRP_GOLDEN_WIN_X[8][40] = {matrix(x_ref)};
#define ERRP_GOLDEN_WIN_LDA {f32(lda_ref)}

// IIR: entrada (DC grande + deriva + 6 Hz) y salida de scipy sosfilt con zi * x[0]
#define ERRP_GOLDEN_IIR_LEN {len(x_iir)}
static const float ERRP_GOLDEN_IIR_X[ERRP_GOLDEN_IIR_LEN] = {farr(x_iir)};
static const float ERRP_GOLDEN_IIR_Y[ERRP_GOLDEN_IIR_LEN] = {farr(y_iir)};
""")
    print(f"gate {gate_uv:g} µV / flat {flat_uv:g} µV / gyro {gate_gyro:g} °/s, LDA AUC "
          f"{lda['evaluation']['auc']:.3f} -> {FW.relative_to(REPO)}/include/errp_params.h, errp_golden.h")


if __name__ == "__main__":
    main()
