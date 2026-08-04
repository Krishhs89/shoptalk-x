"""Integration tests for how main.py wires the semantic caches into
/search/text -- the two safety-critical properties are: (1) a repeated
query on a FRESH session skips retrieval, rerank, AND the LLM step
entirely, and (2) a repeated query as a FOLLOW-UP (non-empty history) only
ever reuses the retrieval cache, never the cached LLM answer, since the
answer must account for conversation context that didn't exist when it was
first cached. Retrieval/rerank/LLM are all monkeypatched to call-counters
so this runs in seconds without loading real models or hitting Ollama."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shoptalk.api.logging_store as logging_store
import shoptalk.api.main as main

client = TestClient(main.app)

_HIT = {
    "item_id": "B000TEST",
    "metadata": {"item_name": "Test Widget", "category": "TEST", "brand": "Acme", "price_usd": 9.99},
    "document": "a test widget",
    "score": 0.9,
    "rerank_score": 1.0,
}


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_store, "DB_PATH", tmp_path / "test.db")
    main._state["cfg"] = {
        "embeddings": {"text_model": "fake-model"},
        "retrieval": {"stage1_k": 100, "top_k": 10},
        "llm": {"context_products": 8, "conversation_max_turns": 6},
        "cache": {"enabled": True, "similarity_threshold": 0.93, "ttl_seconds": 600, "max_entries": 500},
    }
    main._state["sessions"] = {}
    main._state["session_users"] = {}
    main._caches["retrieval"] = None
    main._caches["full_response"] = None

    calls = {"stage1": 0, "rerank": 0, "llm": 0}

    def fake_stage1_search(query, top_k, cfg):
        calls["stage1"] += 1
        return [_HIT]

    def fake_rerank(query, hits, top_k, cfg):
        calls["rerank"] += 1
        return hits

    def fake_answer_from_hits(query, hits, history, cfg, stream):
        calls["llm"] += 1
        return f"answer for: {query}"

    # Deterministic fake embeddings: same base direction for near-duplicate
    # phrasings of the same intent, orthogonal direction for anything else.
    def fake_embed_query(query, model_name):
        lowered = query.lower()
        if "phone case" in lowered or "phone cover" in lowered:
            return _unit([1.0, 0.02, 0.0]) if "cover" in lowered else _unit([1.0, 0.0, 0.0])
        return _unit([0.0, 1.0, 0.0])

    monkeypatch.setattr(main, "stage1_search", fake_stage1_search)
    monkeypatch.setattr(main, "rerank", fake_rerank)
    monkeypatch.setattr(main, "answer_from_hits", fake_answer_from_hits)
    monkeypatch.setattr(main, "embed_query", fake_embed_query)

    yield calls

    main._state["cfg"] = None


def _search(query, session_id=None, user_name=None):
    body = {"query": query, "stream": False}
    if session_id:
        body["session_id"] = session_id
    if user_name:
        body["user_name"] = user_name
    resp = client.post("/search/text", json=body)
    assert resp.status_code == 200
    return resp.json()


def test_repeated_query_on_fresh_sessions_hits_full_response_cache(_isolated_environment):
    calls = _isolated_environment
    first = _search("red phone case under $30")
    assert calls == {"stage1": 1, "rerank": 1, "llm": 1}

    second = _search("red phone case under $30")  # a different (fresh) session
    assert calls == {"stage1": 1, "rerank": 1, "llm": 1}  # nothing incremented -- full hit
    assert second["answer"] == first["answer"]
    assert second["latency"]["total_ms"] == 0.0
    assert second["session_id"] != first["session_id"]  # still a distinct conversation


def test_near_duplicate_phrasing_hits_full_response_cache(_isolated_environment):
    calls = _isolated_environment
    _search("red phone case under $30")
    _search("red phone cover under $30")
    assert calls == {"stage1": 1, "rerank": 1, "llm": 1}


def test_followup_turn_only_reuses_retrieval_cache_not_the_answer(_isolated_environment):
    """The safety-critical case: history must never be short-circuited by a
    cached answer that was generated without it."""
    calls = _isolated_environment
    first = _search("red phone case under $30")
    session_id = first["session_id"]

    followup = _search("red phone case under $30", session_id=session_id)
    # retrieval was reused (stage1/rerank NOT incremented again)...
    assert calls["stage1"] == 1
    assert calls["rerank"] == 1
    # ...but the LLM WAS invoked again, because this turn has prior history
    assert calls["llm"] == 2
    assert followup["answer"] == "answer for: red phone case under $30"


def test_query_with_different_number_does_not_share_cache_entry(_isolated_environment):
    calls = _isolated_environment
    _search("red phone case under $30")
    _search("red phone case under $100")
    # numeric signature differs -- must be treated as a fresh query end to end
    assert calls == {"stage1": 2, "rerank": 2, "llm": 2}


def test_cache_disabled_via_config_falls_back_to_always_computing(_isolated_environment):
    calls = _isolated_environment
    main._state["cfg"]["cache"]["enabled"] = False
    _search("red phone case under $30")
    _search("red phone case under $30")
    assert calls == {"stage1": 2, "rerank": 2, "llm": 2}
