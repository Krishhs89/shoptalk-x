"""Loads configs/config.yaml into a plain dict."""
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
    if "eval" in cfg:
        cfg["eval"]["golden_set_path"] = str(REPO_ROOT / cfg["eval"]["golden_set_path"])
        cfg["eval"]["results_dir"] = str(REPO_ROOT / cfg["eval"]["results_dir"])
    return cfg
