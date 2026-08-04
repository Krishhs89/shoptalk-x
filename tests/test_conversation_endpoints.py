"""Endpoint-level tests for the new GET /conversations and
GET /conversations/{session_id} routes (per-user search history + resume).
Uses TestClient WITHOUT entering it as a context manager, which skips the
app's lifespan handler -- these routes only touch logging_store, not the
ML models the lifespan warms, so this keeps the test fast (no model
downloads/loads) while still exercising real routing/dependency wiring."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shoptalk.api.logging_store as logging_store
from shoptalk.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_store, "DB_PATH", tmp_path / "test.db")


def test_list_conversations_empty_for_unknown_user():
    resp = client.get("/conversations", params={"user_name": "nobody"})
    assert resp.status_code == 200
    assert resp.json() == {"conversations": []}


def test_list_conversations_requires_non_blank_user_name():
    resp = client.get("/conversations", params={"user_name": "   "})
    assert resp.status_code == 422


def test_list_and_get_conversation_round_trip():
    logging_store.save_conversation(
        "sess1", "krishna", [{"role": "user", "content": "red shoes"}, {"role": "assistant", "content": "here"}]
    )

    listed = client.get("/conversations", params={"user_name": "krishna"}).json()["conversations"]
    assert len(listed) == 1
    assert listed[0]["session_id"] == "sess1"
    assert listed[0]["preview"] == "red shoes"

    detail = client.get("/conversations/sess1").json()
    assert detail["user_name"] == "krishna"
    assert detail["history"] == [
        {"role": "user", "content": "red shoes"}, {"role": "assistant", "content": "here"}
    ]


def test_get_unknown_conversation_404s():
    resp = client.get("/conversations/does-not-exist")
    assert resp.status_code == 404
