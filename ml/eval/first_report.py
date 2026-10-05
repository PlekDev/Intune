#!/usr/bin/env python3
"""
C4: primer reporte sobre un .npz de ml/data/processed/ (base de la prueba 11).

  python ml/eval/first_report.py ml/data/processed/<sesion>.npz [--out ml/eval]

Escribe report_<sesion>.md y grand_average_<sesion>.png con:
  1. epocas por motivo de rechazo (y por etiqueta),
  2. gran promedio error - correcto en Fz/Cz (picos, t por muestra),
  3. AUC con validacion cruzada de un LDA de contraste (56 rasgos: 7 bins x 8 canales, 160-700 ms)
     entrenado SOLO con epocas limpias fuera de calibracion. No es el baseline_lda.py de C2: sirve
     para saber si el pipeline conserva la senal.
Con la sesion sintetica esto solo prueba plomeria: el ErrP esta puesto a mano.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FS_OUT = 50


def md(df, index=True):
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def bins_features(X):
    # muestras 8..35 = 160..700 ms; 7 bins de 4 muestras (80 ms) por canal
    seg = X[:, :, 8:36].reshape(len(X), X.shape[1], 7, 4).mean(axis=3)
    return seg.reshape(len(X), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    X, y, rej, cal = d["X"], d["y"], d["rejected"], d["is_calibration"]
    reason = d["reject_reason"]
    ch = list(d["ch_names"])
    sess = str(d["session"])
    lat = float(d["latency_offset_ms"])
    t_ms = np.arange(40) * 1000 / FS_OUT  # 0..780 ms (inicio de cada bloque de 20 ms)

    lines = [f"# Primer reporte C4: {sess}", "",
             f"Pipeline `{d['pipeline_version']}`, latency_offset_ms = {lat:g}, gate_uv = {float(d['gate_uv']):g}, "
             f"gate_gyro = {float(d['gate_gyro']):g}. Sujeto: {d['subject']}.", ""]

    # ---- 1. rechazo ----
    prim = np.array([r.split("+")[0] for r in reason])
    lab = np.where(y == 0, "correct", np.where(y == 1, "error", "manual"))
    tab = pd.crosstab(pd.Series(prim, name="motivo (primero)"), pd.Series(lab, name="etiqueta"), margins=True)
    lines += ["## Epocas por motivo de rechazo", "", md(tab), ""]
    allr = pd.Series([m for r in reason for m in r.split("+") if m != "ok"]).value_counts()
    if len(allr):
        lines += ["Conteo contando todos los motivos de cada epoca (una epoca puede tener varios):", "",
                  md(allr.to_frame("epocas")), ""]
    n_ok = int((~rej).sum())
    lines += [f"Epocas limpias: {n_ok} de {len(y)} ({100 * n_ok / len(y):.1f} %). "
              f"Calibracion: {int(d['n_calibration'])} (norm_valid = {bool(d['norm_valid'])}).", ""]

    # ---- 2. gran promedio (limpias, sin calibracion) ----
    use = ~rej & ~cal
    Xc, Xe = X[use & (y == 0)], X[use & (y == 1)]
    lines += [f"## Gran promedio error - correcto (limpias, fuera de calibracion: {len(Xe)} error, {len(Xc)} correct)", ""]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    rows = []
    for ax, name in zip(axs, ["Fz", "Cz"]):
        c = ch.index(name)
        diff = Xe[:, c].mean(0) - Xc[:, c].mean(0)
        se = np.sqrt(Xe[:, c].var(0, ddof=1) / len(Xe) + Xc[:, c].var(0, ddof=1) / len(Xc))
        tstat = diff / se
        neg = (t_ms >= 150) & (t_ms <= 400)
        pos = (t_ms >= 250) & (t_ms <= 550)
        i_n = np.flatnonzero(neg)[np.argmin(diff[neg])]
        i_p = np.flatnonzero(pos)[np.argmax(diff[pos])]
        rows.append(dict(canal=name, neg_ms=int(t_ms[i_n]), neg_uV=round(float(diff[i_n]), 2), neg_t=round(float(tstat[i_n]), 1),
                         pos_ms=int(t_ms[i_p]), pos_uV=round(float(diff[i_p]), 2), pos_t=round(float(tstat[i_p]), 1)))
        ax.plot(t_ms, Xc[:, c].mean(0), label="correct", color="#555")
        ax.plot(t_ms, Xe[:, c].mean(0), label="error", color="#c0392b")
        ax.plot(t_ms, diff, label="error - correct", color="#1f6fb5", lw=2)
        ax.fill_between(t_ms, diff - 1.96 * se, diff + 1.96 * se, color="#1f6fb5", alpha=0.15)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{name} (IC 95 % de la diferencia)")
        ax.set_xlabel("ms desde t = 0 (bloques de 20 ms)")
    axs[0].set_ylabel("uV (baseline [-200, 0))")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    png = a.out / f"grand_average_{sess}.png"
    fig.savefig(png, dpi=130)
    lines += [md(pd.DataFrame(rows), index=False), "",
              "Pico negativo buscado en 150-400 ms y positivo en 250-550 ms de la curva de diferencia; "
              "`*_t` es el t de Welch en ese punto.", "", f"![gran promedio]({png.name})", ""]

    # ---- 3. AUC CV ----
    idx = np.flatnonzero(use & (y >= 0))
    F, yy = bins_features(X[idx]), y[idx]
    p = np.zeros(len(idx))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, yy):
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[tr], yy[tr])
        p[te] = m.decision_function(F[te])
    auc = roc_auc_score(yy, p)
    fa = np.percentile(p[yy == 0], 99)
    lines += ["## AUC (LDA de contraste, CV 5 pliegues dentro de la sesion)", "",
              f"- AUC = **{auc:.3f}** ({int((yy == 1).sum())} error vs {int((yy == 0).sum())} correct).",
              f"- Deteccion de error con 1 % de falsa alarma sobre correct: {100 * np.mean(p[yy == 1] > fa):.1f} %.",
              "- La CV mezcla epocas de una sola sesion: es optimista y solo valida que la senal sobrevive al pipeline. "
              "Para el reporte real, split por sesion/operador (FORMAT.md).", ""]
    lines += ["## Lectura", "",
              "Datos SINTETICOS (make_synthetic.py): el ErrP se inserto a mano. Esto prueba que el pipeline "
              "(ventana, baseline, diezmado, compuerta) lo conserva y que el reporte funciona; "
              "la prueba 11 real requiere grabaciones reales."]
    out = a.out / f"report_{sess}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}\n-> {png}")


if __name__ == "__main__":
    main()
