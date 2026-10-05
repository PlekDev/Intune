"""Exporta el ErrP-AE a TFLite int8 (TFLite Micro en el S3) y valida el score.

    uv run --project ml/autoencoder ml/autoencoder/training/export_tflite.py

Cuantización post-entrenamiento full-integer (entrada y salida int8) con un set
representativo de epochs correctos de train. El score se calcula como en el
S3: epoch normalizado en float -> cuantizar -> invoke -> decuantizar ->
MSE en float contra el epoch normalizado. Criterio: correlación de Pearson
entre score float (Keras) e int8 >= 0.98 sobre test (correctos + errores).
Si no pasa, sale con código 1.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

AE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AE_DIR))
sys.path.insert(0, str(AE_DIR / "training"))
import errp_pipeline as ep  # noqa: E402
from train import META_PATH, MODEL_PATH, load_splits  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

TFLITE_PATH = AE_DIR / "models" / "errp_ae_int8.tflite"
MIN_CORR = 0.98


def convert(model: keras.Model, rep: np.ndarray) -> bytes:
    # Vía SavedModel: from_concrete_functions deja READ_VARIABLE sin congelar con Keras 3.
    # Batch fijo en 1 (lo que corre el S3).
    with tempfile.TemporaryDirectory() as tmp:
        archive = keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint("serve", lambda x: model(x, training=False),
                             input_signature=[tf.TensorSpec([1, *model.input_shape[1:]], tf.float32)])
        archive.write_out(tmp, verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp, signature_keys=["serve"])
        return _quantize(conv, rep)


def _quantize(conv, rep: np.ndarray) -> bytes:
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([r[None]] for r in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class Int8Scorer:
    def __init__(self, tflite: bytes):
        # Sin XNNPACK: kernels de referencia, más cercanos a TFLite Micro y lista de ops real
        self.it = tf.lite.Interpreter(
            model_content=tflite,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.in_q = self.inp["quantization"]
        self.out_q = self.out["quantization"]

    def reconstruct(self, e: np.ndarray) -> np.ndarray:
        s_in, z_in = self.in_q
        s_out, z_out = self.out_q
        rec = np.empty_like(e)
        for i in range(len(e)):
            q = np.clip(np.round(e[i] / s_in) + z_in, -128, 127).astype(np.int8)
            self.it.set_tensor(self.inp["index"], q[None])
            self.it.invoke()
            rec[i] = (self.it.get_tensor(self.out["index"])[0].astype(np.float32) - z_out) * s_out
        return rec

    def score(self, e: np.ndarray) -> np.ndarray:
        return ep.mse_score(e, self.reconstruct(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-rep", type=int, default=300, help="epochs del set representativo")
    args = ap.parse_args()

    meta = json.loads(META_PATH.read_text())
    model = keras.models.load_model(MODEL_PATH)
    _, train, _, s_test, test_idx, _ = load_splits(meta["data"], meta["split"]["test_session"])
    rng = np.random.default_rng(0)
    rep_idx = rng.permutation(len(train))[:args.n_rep]
    e_rep = ep.to_model(train.z[rep_idx])
    # Test: acciones evaluadas no rechazadas (correctas y errores) + calibración de la sesión
    idx = np.concatenate([np.flatnonzero(s_test.cal_mask), test_idx[~s_test.rejected[test_idx]]])
    e_test = ep.to_model(s_test.z(idx))

    tflite = convert(model, e_rep)
    TFLITE_PATH.write_bytes(tflite)

    q = Int8Scorer(tflite)
    s_float = ep.mse_score(e_test, model.predict(e_test, verbose=0))
    s_int8 = q.score(e_test)
    corr = float(np.corrcoef(s_float, s_int8)[0, 1])
    ok = corr >= MIN_CORR
    ops = sorted({op["op_name"] for op in q.it._get_ops_details()})

    meta["tflite"] = {
        "file": TFLITE_PATH.name, "bytes": len(tflite), "ops": ops,
        "input": {"dtype": "int8", "shape": [int(v) for v in q.inp["shape"]],
                  "scale": float(q.in_q[0]), "zero_point": int(q.in_q[1])},
        "output": {"dtype": "int8", "shape": [int(v) for v in q.out["shape"]],
                   "scale": float(q.out_q[0]), "zero_point": int(q.out_q[1])},
        "float_int8_corr": corr, "corr_pass": ok, "n_test": len(e_test),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{TFLITE_PATH.name}: {len(tflite)} B, ops {ops}")
    print(f"entrada int8 scale {q.in_q[0]:.5f} zp {q.in_q[1]}, salida scale {q.out_q[0]:.5f} zp {q.out_q[1]}")
    print(f"correlación score float vs int8: {corr:.4f} ({'OK' if ok else 'FALLA'}, mínimo {MIN_CORR})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
