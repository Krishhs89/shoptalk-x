"""Tests for embed_image.py's resume-from-checkpoint logic -- same pattern
and same motivation as test_caption_checkpoint.py: a Colab disconnect
partway through CLIP-embedding a 10k-image catalog used to mean losing
every embedding computed so far, since the old code only wrote to Chroma
once, after the full loop finished. Full CLIP-inference resumability was
verified manually against the real local Chroma collections (see commit
message); this covers the pure checkpoint-file parsing logic."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.embeddings.embed_image import _load_checkpoint


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert _load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_parses_entries(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"item_id": "A1", "embedding": [0.1, 0.2]}) + "\n"
        + json.dumps({"item_id": "A2", "embedding": [0.3, 0.4]}) + "\n"
    )
    assert _load_checkpoint(ckpt) == {"A1": [0.1, 0.2], "A2": [0.3, 0.4]}


def test_load_checkpoint_skips_blank_lines(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(json.dumps({"item_id": "A1", "embedding": [1.0]}) + "\n\n\n")
    assert _load_checkpoint(ckpt) == {"A1": [1.0]}


def test_load_checkpoint_last_write_wins_for_duplicate_item_id(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"item_id": "A1", "embedding": [1.0]}) + "\n"
        + json.dumps({"item_id": "A1", "embedding": [2.0]}) + "\n"
    )
    assert _load_checkpoint(ckpt) == {"A1": [2.0]}
