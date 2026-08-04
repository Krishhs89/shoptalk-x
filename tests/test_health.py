"""Guards a real gap found during the effectiveness/efficiency review: the
old /health handler returned status="ok" unconditionally from static config
values, never actually checking whether Ollama (a separate container/process
that can die independently after startup) was reachable. That let the UI's
sidebar keep showing "API online" while every search silently failed at the
LLM step. Verifies the endpoint now reports "degraded" and flags the LLM
entry when Ollama can't be reached, without needing real models loaded
(TestClient without a `with` block skips the model-warming lifespan)."""
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shoptalk.api.main as main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _fake_cfg():
    main._state["cfg"] = {
        "embeddings": {"text_model": "bge-small"},
        "retrieval": {"rerank_model": "ce-model"},
        "clip": {"model_name": "ViT-B-32", "pretrained": "laion2b"},
        "captioning": {"model": "blip-base"},
        "llm": {"model": "llama3.1:8b", "base_url": "http://ollama-test:11434"},
    }
    yield
    main._state["cfg"] = None


class _FakeResponse:
    def __init__(self, ok):
        self.ok = ok


def test_health_ok_when_ollama_reachable(monkeypatch):
    monkeypatch.setattr(main.requests, "get", lambda url, timeout: _FakeResponse(ok=True))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "UNREACHABLE" not in body["models_loaded"]["llm"]


def test_health_degraded_when_ollama_unreachable(monkeypatch):
    def _raise(url, timeout):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(main.requests, "get", _raise)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "UNREACHABLE" in body["models_loaded"]["llm"]
