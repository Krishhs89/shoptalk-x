"""Tests for the SQLite-backed conversation persistence added so a user can
see and resume their past searches ("show the history for each user for
each search ... continue the conversation"). Verifies round-trip save/load,
the MAX_STORED_TURNS cap, and that list_conversations returns a lightweight
per-user preview rather than full history."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shoptalk.api.logging_store as logging_store


def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_store, "DB_PATH", tmp_path / "test.db")


def test_save_and_load_round_trip(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    logging_store.save_conversation("sess1", "krishna", history)

    loaded = logging_store.load_conversation("sess1")
    assert loaded["user_name"] == "krishna"
    assert loaded["history"] == history


def test_load_unknown_session_returns_none(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    assert logging_store.load_conversation("nope") is None


def test_save_upserts_same_session(monkeypatch, tmp_path):
    """A conversation is saved after every turn, not just once -- must
    overwrite in place rather than accumulate duplicate rows."""
    _use_tmp_db(monkeypatch, tmp_path)
    logging_store.save_conversation("sess1", "krishna", [{"role": "user", "content": "first"}])
    logging_store.save_conversation(
        "sess1", "krishna", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}]
    )
    loaded = logging_store.load_conversation("sess1")
    assert len(loaded["history"]) == 2
    assert len(logging_store.list_conversations("krishna")) == 1


def test_save_conversation_caps_stored_turns(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    long_history = [{"role": "user", "content": f"turn {i}"} for i in range(300)]
    logging_store.save_conversation("sess1", "krishna", long_history)
    loaded = logging_store.load_conversation("sess1")
    assert len(loaded["history"]) == logging_store.MAX_STORED_TURNS
    assert loaded["history"][-1]["content"] == "turn 299"  # keeps the most recent turns, not the oldest


def test_list_conversations_scoped_to_user_and_ordered_by_recency(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    counter = iter([100.0, 200.0, 300.0])
    monkeypatch.setattr(logging_store.time, "time", lambda: next(counter))

    logging_store.save_conversation("sess_a", "krishna", [{"role": "user", "content": "red shoes"}])
    logging_store.save_conversation("sess_b", "krishna", [{"role": "user", "content": "blue shirt"}])
    logging_store.save_conversation("sess_c", "someone_else", [{"role": "user", "content": "green hat"}])

    results = logging_store.list_conversations("krishna")
    session_ids = [r["session_id"] for r in results]
    assert session_ids == ["sess_b", "sess_a"]  # most-recently-updated first
    assert "sess_c" not in session_ids  # not this user's conversation


def test_list_conversations_preview_uses_first_user_turn(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    history = [
        {"role": "assistant", "content": "should not appear in preview"},
        {"role": "user", "content": "looking for a red dress"},
    ]
    logging_store.save_conversation("sess1", "krishna", history)
    results = logging_store.list_conversations("krishna")
    assert results[0]["preview"] == "looking for a red dress"
    assert results[0]["turn_count"] == 2


def test_list_conversations_truncates_long_preview(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    logging_store.save_conversation("sess1", "krishna", [{"role": "user", "content": "x" * 200}])
    preview = logging_store.list_conversations("krishna")[0]["preview"]
    assert len(preview) == 83  # 80 chars + "..."
    assert preview.endswith("...")
