"""
Model class definitions and weight loaders for the trained Graph VAE and Geometry
Predictor. Imports are deferred so the rest of the app still runs without torch.
"""

from __future__ import annotations
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GRAPH_VAE_PATH = MODELS_DIR / "graph_vae.pth"
GEOMETRY_PATH = MODELS_DIR / "geometry_predictor.pth"


# ── Shared GCN layer (same as training) ───────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        self.fc = nn.Linear(in_d, out_d)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_hat = A + torch.eye(A.size(0), device=A.device)
        deg = A_hat.sum(1)
        D = torch.diag((deg + 1e-6).pow(-0.5))
        return F.relu((D @ A_hat @ D) @ self.fc(H))


# ── Graph VAE (Cell B v2: 3 GCN layers 128→256→128, latent 16) ────────────────

class GraphVAE(nn.Module):
    def __init__(self, num_classes: int, latent: int = 16):
        super().__init__()
        self.gcn1 = GCNLayer(num_classes, 128)
        self.gcn2 = GCNLayer(128, 256)
        self.gcn3 = GCNLayer(256, 128)
        self.mu = nn.Linear(128, latent)
        self.var = nn.Linear(128, latent)
        self.dec_nodes = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def encode(self, x, A):
        h = self.gcn3(self.gcn2(self.gcn1(x, A), A), A)
        return self.mu(h), self.var(h)

    def decode(self, z):
        node_logits = self.dec_nodes(z)
        edge_logits = z @ z.T
        return node_logits, edge_logits

    def forward(self, x, A):
        mu, log_var = self.encode(x, A)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        node_logits, edge_logits = self.decode(z)
        return node_logits, edge_logits, mu, log_var


# ── Geometry Predictor (Cell D1 v2: 4 GCN layers 128→256→256→128 + MLP head) ──

class GeometryPredictor(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.gcn1 = GCNLayer(num_classes, 128)
        self.gcn2 = GCNLayer(128, 256)
        self.gcn3 = GCNLayer(256, 256)
        self.gcn4 = GCNLayer(256, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid(),
        )

    def forward(self, x, A):
        h = self.gcn4(self.gcn3(self.gcn2(self.gcn1(x, A), A), A), A)
        return self.head(h)


# ── Weight loading ────────────────────────────────────────────────────────────

def models_available() -> bool:
    return GRAPH_VAE_PATH.is_file() and GEOMETRY_PATH.is_file()


def load_models(device: str | None = None):
    """Load both trained models. Returns (gcn, geo, all_classes, device).
    Raises FileNotFoundError if weights aren't on disk yet.
    """
    if not models_available():
        missing = [str(p) for p in (GRAPH_VAE_PATH, GEOMETRY_PATH) if not p.is_file()]
        raise FileNotFoundError(
            "Trained model weights missing: " + ", ".join(missing)
            + "\nPlace them under the models/ folder."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gcn_ckpt = torch.load(GRAPH_VAE_PATH, map_location=device, weights_only=False)
    geo_ckpt = torch.load(GEOMETRY_PATH, map_location=device, weights_only=False)

    # Both checkpoints store {'state_dict': ..., 'classes': ALL_CLASSES}
    classes = gcn_ckpt.get("classes") or geo_ckpt.get("classes")
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes' list — re-export from training.")
    num_classes = len(classes)

    gcn = GraphVAE(num_classes).to(device)
    gcn.load_state_dict(gcn_ckpt["state_dict"])
    gcn.eval()

    geo = GeometryPredictor(num_classes).to(device)
    geo.load_state_dict(geo_ckpt["state_dict"])
    geo.eval()

    return gcn, geo, classes, device
