"""Loads configs/config.yaml into a plain dict."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # resolve data dirs relative to repo root so scripts work from anywhere
    cfg["data"]["raw_dir"] = str(REPO_ROOT / cfg["data"]["raw_dir"])
    cfg["data"]["processed_dir"] = str(REPO_ROOT / cfg["data"]["processed_dir"])
    cfg["embeddings"]["chroma_dir"] = str(REPO_ROOT / cfg["embeddings"]["chroma_dir"])
    if "clip" in cfg:
        cfg["clip"]["chroma_dir"] = str(REPO_ROOT / cfg["clip"]["chroma_dir"])
    if "eval" in cfg:
        cfg["eval"]["golden_set_path"] = str(REPO_ROOT / cfg["eval"]["golden_set_path"])
        cfg["eval"]["results_dir"] = str(REPO_ROOT / cfg["eval"]["results_dir"])
    if "llm" in cfg:
        cfg["llm"]["base_url"] = os.environ.get("OLLAMA_BASE_URL", cfg["llm"]["base_url"])
        cfg["llm"]["model"] = os.environ.get("OLLAMA_MODEL", cfg["llm"]["model"])
    if "verification" in cfg:
        cfg["verification"]["pairs_path"] = str(REPO_ROOT / cfg["verification"]["pairs_path"])
        cfg["verification"]["model_path"] = str(REPO_ROOT / cfg["verification"]["model_path"])
    if "counting" in cfg:
        cfg["counting"]["weights_dir"] = str(REPO_ROOT / cfg["counting"]["weights_dir"])
    if "personalization" in cfg:
        cfg["personalization"]["interactions_path"] = str(
            REPO_ROOT / cfg["personalization"]["interactions_path"]
        )
    return cfg
