"""Shared MLP architecture + pairwise featurization for the verification head
(design doc §3.3b). Imported by both train_verification.py and verify.py so
training and inference can never drift out of sync."""
import numpy as np
import torch
from torch import nn


def pair_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Concatenates both embeddings with their absolute difference and
    elementwise product -- standard featurization for pairwise metric-learning
    classifiers (lets the MLP learn both similarity and directional cues)."""
    return np.concatenate([emb_a, emb_b, np.abs(emb_a - emb_b), emb_a * emb_b], axis=-1)


class VerificationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits
