"""Tests for finetune_text.py's per-epoch resume state -- the third
Colab-disconnect fix: a disconnect during epoch 3/3 of fine-tuning used to
lose all training because the model was only saved after model.fit()
finished every epoch. Covers the pure state-file logic (load/save/guards);
actual one-epoch-at-a-time training is validated on the real pipeline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.embeddings.finetune_text import _load_state, _save_state


def test_load_state_missing_file_returns_zero(tmp_path):
    assert _load_state(tmp_path / "_finetune_state.json") == 0


def test_save_then_load_roundtrip(tmp_path):
    state_path = tmp_path / "_finetune_state.json"
    # the guard requires the saved model's config.json to exist
    (tmp_path / "config.json").write_text("{}")
    _save_state(state_path, completed_epochs=2, total_epochs=3)
    assert _load_state(state_path) == 2


def test_load_state_distrusts_state_without_saved_model(tmp_path):
    """State file says 2 epochs done, but no model config.json -> half-written
    save from a crash mid-save; must restart from scratch, not load garbage."""
    state_path = tmp_path / "_finetune_state.json"
    state_path.write_text(json.dumps({"completed_epochs": 2, "total_epochs": 3}))
    assert _load_state(state_path) == 0


def test_load_state_corrupt_json_returns_zero(tmp_path):
    state_path = tmp_path / "_finetune_state.json"
    state_path.write_text("{not json")
    assert _load_state(state_path) == 0


def test_save_state_is_atomic_no_tmp_left_behind(tmp_path):
    state_path = tmp_path / "_finetune_state.json"
    (tmp_path / "config.json").write_text("{}")
    _save_state(state_path, 1, 3)
    assert not state_path.with_suffix(".json.tmp").exists()
    assert _load_state(state_path) == 1
