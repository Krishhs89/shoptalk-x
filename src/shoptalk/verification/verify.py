"""
Inference-side of the verification head: given a received-item photo and an
order's item_id, decide match / mismatch / suspect (design doc §3.3b, §8
"human-review path for verification 'suspect' verdicts").

A band around the trained threshold maps to "suspect" rather than forcing a
binary call -- this is the calibration/human-in-the-loop routing the design
doc requires before any counterfeit accusation.

Usage:
  python -m shoptalk.verification.verify --image photo.jpg --order-item-id B07XYZ1234
"""
import argparse
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image

from shoptalk.config import load_config
from shoptalk.verification.model import VerificationMLP, pair_features

SUSPECT_MARGIN = 0.15  # +/- around threshold, in probability space, routed to human review

_clip_model = None
_clip_preprocess = None
_clip_device = None
_mlp_model = None
_mlp_checkpoint = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_clip(vcfg: dict):
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        _clip_device = _resolve_device(vcfg["device"])
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            vcfg["clip_model_name"], pretrained=vcfg["clip_pretrained"], device=_clip_device
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_device


def _get_mlp(vcfg: dict):
    global _mlp_model, _mlp_checkpoint
    if _mlp_model is None:
        checkpoint = torch.load(vcfg["model_path"], map_location="cpu", weights_only=False)
        model = VerificationMLP(input_dim=checkpoint["input_dim"], hidden_dim=checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _mlp_model, _mlp_checkpoint = model, checkpoint
    return _mlp_model, _mlp_checkpoint


def _embed(image_path: str, clip_model, preprocess, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def verify_photo(image_path: str, order_item_id: str, cfg: dict = None) -> dict:
    cfg = cfg or load_config()
    vcfg = cfg["verification"]

    products_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    row = products_df[products_df["item_id"] == order_item_id]
    if row.empty or not row.iloc[0]["image_available"]:
        raise ValueError(f"no catalog image available for order_item_id={order_item_id!r}")
    # Rebuild the path from image_id + THIS process's raw_dir rather than
    # trusting the stored `image_path` column: that column is an absolute
    # path baked in wherever preprocess.py last ran (e.g. the host Mac), and
    # this endpoint can be serving from a different filesystem root (a
    # Docker container mounting the same data/ at a different absolute
    # path -- caught via live testing: FileNotFoundError on the container's
    # /verify calls even though the file was right there under a different
    # prefix).
    catalog_image_path = str(Path(cfg["data"]["raw_dir"]) / "images" / f"{row.iloc[0]['image_id']}.jpg")

    clip_model, preprocess, device = _get_clip(vcfg)
    received_emb = _embed(image_path, clip_model, preprocess, device)
    catalog_emb = _embed(catalog_image_path, clip_model, preprocess, device)

    mlp_model, checkpoint = _get_mlp(vcfg)
    features = pair_features(received_emb, catalog_emb)
    with torch.no_grad():
        logit = mlp_model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
        confidence = torch.sigmoid(logit).item()

    threshold = checkpoint["threshold"]
    if confidence >= threshold + SUSPECT_MARGIN:
        verdict = "match"
    elif confidence <= threshold - SUSPECT_MARGIN:
        verdict = "mismatch"
    else:
        verdict = "suspect"  # routed to human review, never an automatic accusation

    return {"verdict": verdict, "confidence": confidence, "threshold": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--order-item-id", required=True)
    args = parser.parse_args()

    result = verify_photo(args.image, args.order_item_id)
    print(f"order_item_id: {args.order_item_id}")
    print(f"verdict: {result['verdict']}  (confidence={result['confidence']:.4f}, threshold={result['threshold']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
