"""Tests for caption_images.py's resume-from-checkpoint logic, added after
a real Colab runtime disconnect mid-captioning-run wiped hours of progress:
the script used to hold all captions in memory and write output only once,
at the very end, so any interruption lost everything completed so far. Full
BLIP-inference resumability was verified manually (a fake partial checkpoint
+ real run, see commit message); this covers the pure checkpoint-file
parsing logic that a live model isn't needed for."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.data.caption_images import _load_checkpoint


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert _load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_parses_entries(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"item_id": "A1", "caption": "a red shoe"}) + "\n"
        + json.dumps({"item_id": "A2", "caption": "a blue hat"}) + "\n"
    )
    assert _load_checkpoint(ckpt) == {"A1": "a red shoe", "A2": "a blue hat"}


def test_load_checkpoint_skips_blank_lines(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(json.dumps({"item_id": "A1", "caption": "x"}) + "\n\n\n")
    assert _load_checkpoint(ckpt) == {"A1": "x"}


def test_load_checkpoint_last_write_wins_for_duplicate_item_id(tmp_path):
    """A batch that was appended twice (e.g. a crash right after the flush
    but before the process could move on) should resolve to the latest
    entry, not error or silently keep the first."""
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"item_id": "A1", "caption": "first"}) + "\n"
        + json.dumps({"item_id": "A1", "caption": "second"}) + "\n"
    )
    assert _load_checkpoint(ckpt) == {"A1": "second"}
