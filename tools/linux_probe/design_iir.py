#!/usr/bin/env python3
"""
Genera firmware/common/eeg_iir.h: band-pass causal 1–15 Hz (regla dura 1 de CLAUDE.md).

    uv run --with scipy python tools/linux_probe/design_iir.py

Diseño: scipy.signal.butter(2, [1, 15], btype='band', fs=250, output='sos')
(pasa-banda de 4º orden, 2 secciones). C4 debe filtrar offline con estos mismos
coeficientes y de forma causal: design_sos() + scipy.signal.sosfilt.
"""

from pathlib import Path

import numpy as np
from scipy import signal

FS = 250.0
BAND = (1.0, 15.0)
N = 2  # butter(N) pasa-banda -> orden 2N

OUT = Path(__file__).resolve().parents[2] / "firmware" / "common" / "eeg_iir.h"


def design_sos():
    return signal.butter(N, BAND, btype="band", fs=FS, output="sos")


def main():
    sos = design_sos()
    zi = signal.sosfilt_zi(sos)  # estado estacionario para un escalón unitario
    f = np.array([2, 4, 6, 8, 10, 12])
    b, a = signal.sos2tf(sos)
    _, gd = signal.group_delay((b, a), w=f, fs=FS)
    gd_txt = ", ".join(f"{int(fr)} Hz {g / FS * 1000:.0f} ms" for fr, g in zip(f, gd))

    def row(v):
        return "{" + ", ".join(f"{x:.17g}" for x in v) + "}"

    sos_rows = ",\n    ".join(row(s) for s in sos)
    zi_rows = ",\n    ".join(row(z) for z in zi)
    n_sec = sos.shape[0]

    OUT.write_text(f"""\
// GENERADO por tools/linux_probe/design_iir.py. No editar a mano.
// Band-pass causal Butterworth {BAND[0]:g}–{BAND[1]:g} Hz, orden {2 * N}, fs = {FS:g} Hz.
// scipy.signal.butter({N}, [{BAND[0]:g}, {BAND[1]:g}], btype='band', fs={FS:g}, output='sos')
// Retardo de grupo: {gd_txt}.
//
// Forma directa II transpuesta, igual que scipy.signal.sosfilt, en double: la señal
// cruda trae offsets de cientos de miles de µV y el pasa-altos de 1 Hz tiene polos
// cerca del círculo unidad; en float el error de redondeo se nota.
// eeg_iir_reset(x0) arranca en estado estacionario para x0 (= sosfilt_zi(sos) * x0),
// así el offset no provoca un transitorio de varios segundos.
#pragma once

#define EEG_IIR_N_SEC {n_sec}
#define EEG_IIR_SETTLE_SAMPLES 500  // 2 s marcado SETTLING tras cada reinicio (regla dura 3)

// Cada fila: b0 b1 b2 a0 a1 a2 (a0 = 1)
static const double EEG_IIR_SOS[EEG_IIR_N_SEC][6] = {{
    {sos_rows}
}};

// Estado estacionario por sección para entrada constante 1
static const double EEG_IIR_ZI[EEG_IIR_N_SEC][2] = {{
    {zi_rows}
}};

typedef struct {{
    double z[EEG_IIR_N_SEC][2];
}} eeg_iir_t;

static inline void eeg_iir_reset(eeg_iir_t *f, double x0)
{{
    for (int s = 0; s < EEG_IIR_N_SEC; s++) {{
        f->z[s][0] = EEG_IIR_ZI[s][0] * x0;
        f->z[s][1] = EEG_IIR_ZI[s][1] * x0;
    }}
}}

static inline double eeg_iir_step(eeg_iir_t *f, double x)
{{
    for (int s = 0; s < EEG_IIR_N_SEC; s++) {{
        const double *c = EEG_IIR_SOS[s];
        double y = c[0] * x + f->z[s][0];
        f->z[s][0] = c[1] * x - c[4] * y + f->z[s][1];
        f->z[s][1] = c[2] * x - c[5] * y;
        x = y;
    }}
    return x;
}}
""")
    print(f"escrito {OUT}")
    print(f"retardo de grupo: {gd_txt}")


if __name__ == "__main__":
    main()
