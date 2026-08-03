import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.config import REPO_ROOT, load_config


def test_load_config_returns_all_top_level_sections():
    cfg = load_config()
    for section in ["data", "embeddings", "captioning", "clip", "retrieval", "eval", "mlflow", "llm", "verification"]:
        assert section in cfg


def test_load_config_resolves_paths_absolute():
    cfg = load_config()
    assert Path(cfg["data"]["raw_dir"]).is_absolute()
    assert Path(cfg["data"]["processed_dir"]).is_absolute()
    assert Path(cfg["embeddings"]["chroma_dir"]).is_absolute()


def test_load_config_paths_under_repo_root():
    cfg = load_config()
    assert Path(cfg["data"]["raw_dir"]).is_relative_to(REPO_ROOT)


def test_ollama_base_url_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example-override:1234")
    cfg = load_config()
    assert cfg["llm"]["base_url"] == "http://example-override:1234"


def test_ollama_model_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "some-other-model")
    cfg = load_config()
    assert cfg["llm"]["model"] == "some-other-model"
