"""ErrP-AE del Detector (C2): autoencoders Keras sobre epochs [8, 40, 1].

conv  (principal, tipo EEGNet, ~49 k parámetros):
    Conv2D temporal 8 filtros (1, 9) same + BN + ReLU     -> 8 x 40 x 8
    DepthwiseConv2D espacial (8, 1), depth mult 2 + BN + ReLU -> 1 x 40 x 16
    AvgPool (1, 2)                                         -> 1 x 20 x 16
    Flatten + Dense 16 (latente lineal)
    Dense 128 ReLU -> Dense 320 -> Reshape 8 x 40 x 1
dense (respaldo si el conv sobreajusta, ~43 k): 320 -> 64 -> 16 -> 64 -> 320.

Solo usa ops soportadas por TFLite Micro (Conv2D, DepthwiseConv2D,
AveragePool2D, FullyConnected, ReLU, Reshape); el converter pliega BN en las convs.
"""
import keras
from keras import layers

INPUT_SHAPE = (8, 40, 1)
LATENT = 16


def build_conv() -> keras.Model:
    x_in = keras.Input(INPUT_SHAPE, name="epoch")
    x = layers.Conv2D(8, (1, 9), padding="same", use_bias=False, name="temporal")(x_in)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D((8, 1), depth_multiplier=2, padding="valid", use_bias=False, name="spatial")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D((1, 2), name="pool")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(LATENT, name="latent")(x)
    x = layers.Dense(128, activation="relu", name="dec1")(z)
    x = layers.Dense(8 * 40, name="dec2")(x)
    x_out = layers.Reshape(INPUT_SHAPE, name="recon")(x)
    return keras.Model(x_in, x_out, name="errp_ae_conv")


def build_dense() -> keras.Model:
    x_in = keras.Input(INPUT_SHAPE, name="epoch")
    x = layers.Flatten()(x_in)
    x = layers.Dense(64, activation="relu", name="enc1")(x)
    z = layers.Dense(LATENT, name="latent")(x)
    x = layers.Dense(64, activation="relu", name="dec1")(z)
    x = layers.Dense(8 * 40, name="dec2")(x)
    x_out = layers.Reshape(INPUT_SHAPE, name="recon")(x)
    return keras.Model(x_in, x_out, name="errp_ae_dense")


BUILDERS = {"conv": build_conv, "dense": build_dense}
