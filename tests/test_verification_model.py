import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.verification.model import VerificationMLP, pair_features


def test_pair_features_shape():
    emb_a = np.random.randn(512).astype(np.float32)
    emb_b = np.random.randn(512).astype(np.float32)
    features = pair_features(emb_a, emb_b)
    assert features.shape == (512 * 4,)  # concat[a, b, |a-b|, a*b]


def test_pair_features_identical_embeddings_zero_diff_block():
    emb = np.random.randn(8).astype(np.float32)
    features = pair_features(emb, emb)
    diff_block = features[16:24]  # a, b, |a-b|, a*b -- each block is 8 wide
    assert np.allclose(diff_block, 0.0)


def test_verification_mlp_forward_shape():
    model = VerificationMLP(input_dim=2048, hidden_dim=128)
    x = torch.randn(4, 2048)
    logits = model(x)
    assert logits.shape == (4,)


def test_verification_mlp_output_is_finite():
    model = VerificationMLP(input_dim=64, hidden_dim=16)
    x = torch.randn(2, 64)
    logits = model(x)
    assert torch.isfinite(logits).all()
