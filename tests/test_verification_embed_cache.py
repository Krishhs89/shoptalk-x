"""Tests for train_verification.py's CLIP embedding cache -- the fourth
Colab-disconnect fix: embedding ~2x pairs images was redone from zero on
every retry. Covers the pure cache-file parsing logic."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.verification.train_verification import _load_embed_cache


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert _load_embed_cache(tmp_path / "nope.jsonl") == {}


def test_load_cache_parses_entries_as_float32_arrays(tmp_path):
    cache = tmp_path / "_clip_embed_cache.jsonl"
    cache.write_text(
        json.dumps({"path": "img/a.jpg", "embedding": [0.1, 0.2]}) + "\n"
        + json.dumps({"path": "img/b.jpg", "embedding": [0.3, 0.4]}) + "\n"
    )
    result = _load_embed_cache(cache)
    assert set(result) == {"img/a.jpg", "img/b.jpg"}
    assert result["img/a.jpg"].dtype == np.float32
    np.testing.assert_allclose(result["img/a.jpg"], [0.1, 0.2], rtol=1e-6)


def test_load_cache_skips_blank_lines_and_last_write_wins(tmp_path):
    cache = tmp_path / "_clip_embed_cache.jsonl"
    cache.write_text(
        json.dumps({"path": "img/a.jpg", "embedding": [1.0]}) + "\n\n"
        + json.dumps({"path": "img/a.jpg", "embedding": [2.0]}) + "\n"
    )
    result = _load_embed_cache(cache)
    np.testing.assert_allclose(result["img/a.jpg"], [2.0])
