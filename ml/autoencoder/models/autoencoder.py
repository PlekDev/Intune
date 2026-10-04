"""Autoencoder denso del nodo Detector (C2, ESP32-S3).

Entrada: 40 features por ventana = 8 canales x 5 bandas, ya normalizadas
(z-score con mean/std de los datos normales, ver config.json).
Arquitectura: 40 -> 16 -> 6 -> 16 -> 40, ReLU en capas intermedias y
reconstrucción lineal. Es tan chica a propósito: se exporta a C a mano y
corre en microsegundos en el S3.
"""
import torch
from torch import nn

# Orden de features: canal mayor, banda menor. El firmware debe usar este mismo orden.
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FEATURE_NAMES = [f"{ch}_{band}" for ch in CHANNELS for band in BANDS]

N_FEATURES = len(FEATURE_NAMES)  # 40
HIDDEN = 16
LATENT = 6


class Autoencoder(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN, latent: int = LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),  # reconstrucción lineal, sin activación
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_mse(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MSE por muestra (promedio sobre las 40 features). Es el score de anomalía."""
    model.eval()
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=1)
