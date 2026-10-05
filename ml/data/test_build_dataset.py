"""Pruebas de build_dataset.py con una sesion sintetica corta. Correr: pytest ml/data -q"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_dataset as bd  # noqa: E402


def make(tmp_path, name, latency):
    subprocess.run([sys.executable, str(HERE / "make_synthetic.py"), "--seed", "3", "--actions", "60",
                    "--latency-ms", str(latency), "--name", name, "--out", str(tmp_path)], check=True,
                   capture_output=True)
    return tmp_path / name


def test_latency_offset_compensates(tmp_path):
    a = bd.build_session(make(tmp_path, "a", 0))
    b = bd.build_session(make(tmp_path, "b", 48), latency_ms=48)
    assert np.array_equal(a["X"], b["X"])
    assert np.array_equal(a["rejected"], b["rejected"])


def test_window_baseline_decimation_by_hand(tmp_path):
    d = make(tmp_path, "c", 0)
    out = bd.build_session(d)
    eeg = pd.read_csv(d / "eeg.csv")
    ev = pd.read_csv(d / "events.csv")
    k = int(np.flatnonzero(~out["rejected"])[5])
    row = int(np.flatnonzero(eeg["counter"].to_numpy() == int(ev["evt_counter"][k]))[0])
    w = eeg[[f"f_{c}" for c in bd.CH]].to_numpy()[row - 50:row + 200]
    w = w - w[:50].mean(0)
    ref = np.stack([w[50 + 5 * i:55 + 5 * i].mean(0) for i in range(40)], axis=1)
    assert out["X"].shape[1:] == (8, 40)
    assert np.allclose(out["X"][k], ref, atol=1e-4)


def test_gate_and_calibration(tmp_path):
    out = bd.build_session(make(tmp_path, "d", 0))
    assert out["is_calibration"].sum() == min(bd.N_CAL, ((out["y"] == 0) & ~out["rejected"]).sum())
    assert not np.any(out["is_calibration"] & (out["y"] != 0))
    assert not np.any(out["is_calibration"] & out["rejected"])
    assert np.all(out["reject_reason"][out["rejected"]] != "ok")
    assert out["norm_std"].shape == (8,) and np.all(out["norm_std"] > 0)
