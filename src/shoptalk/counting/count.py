"""
Quantity/count validation: given a received-item photo and a claimed
quantity, checks whether that many instances of the product are actually
visible in the photo -- using a PRETRAINED object detector (Ultralytics
YOLOv8n, COCO weights), not a custom-trained counting model, since there's
no labeled counting dataset for this catalog.

Scope, honestly stated: COCO's 80 classes are generic household/street
objects. Most of this catalog's ~50 fine-grained categories
(CELLULAR_PHONE_CASE, FINERING, HARDWARE_HANDLE, ...) have no COCO
equivalent at all -- for those, this returns verdict="unsupported" rather
than fabricating a count for a class the detector was never trained to
recognize (see coco_classes.py for exactly which categories/keywords are
covered).

Usage:
  python -m shoptalk.counting.count --image photo.jpg --order-item-id B07XYZ1234 --claimed-qty 3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from shoptalk.config import load_config
from shoptalk.counting.coco_classes import resolve_coco_class

_yolo_model = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_yolo(ccfg: dict):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  # heavy import -- deferred so the rest of the API doesn't pay for it at startup

        weights_path = Path(ccfg["weights_dir"]) / ccfg["yolo_model"]
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        # ultralytics auto-downloads the official checkpoint to this exact
        # path on first use if it isn't already there -- no manual download
        # step needed, matching how the other pretrained models here
        # (open_clip, sentence-transformers) resolve their weights.
        _yolo_model = YOLO(str(weights_path))
    return _yolo_model


def count_detections(image_path: str, coco_class: str, cfg: dict) -> tuple[int, float]:
    """Returns (count, mean_confidence) of `coco_class` instances detected
    in the image. mean_confidence is 0.0 when count is 0."""
    ccfg = cfg["counting"]
    model = _get_yolo(ccfg)
    device = _resolve_device(ccfg["device"])

    results = model.predict(
        image_path, device=device, conf=ccfg["conf_threshold"], verbose=False
    )
    class_names = results[0].names  # {class_id: class_name}, from the model itself
    confidences = [
        float(box.conf[0])
        for box in results[0].boxes
        if class_names[int(box.cls[0])] == coco_class
    ]
    if not confidences:
        return 0, 0.0
    return len(confidences), sum(confidences) / len(confidences)


def _compute_verdict(detected_count: int, claimed_qty: int, suspect_margin: int) -> str:
    diff = abs(detected_count - claimed_qty)
    if diff == 0:
        return "match"
    if diff <= suspect_margin:
        return "suspect"  # e.g. occlusion/overlap undercounting -- routed to human review, never an automatic rejection
    return "mismatch"


def verify_quantity(image_path: str, order_item_id: str, claimed_qty: int, cfg: dict = None) -> dict:
    cfg = cfg or load_config()
    ccfg = cfg["counting"]

    products_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    row = products_df[products_df["item_id"] == order_item_id]
    if row.empty:
        raise ValueError(f"no catalog entry for order_item_id={order_item_id!r}")
    item_name, category = row.iloc[0]["item_name"], row.iloc[0]["category"]

    coco_class = resolve_coco_class(item_name, category)
    if coco_class is None:
        return {
            "verdict": "unsupported",
            "claimed_qty": claimed_qty,
            "detected_count": None,
            "matched_class": None,
            "message": (
                f"category {category!r} isn't covered by the pretrained detector's 80 COCO "
                "classes -- quantity can't be automatically validated for this product; "
                "route to manual review if a count dispute needs resolving."
            ),
        }

    detected_count, mean_conf = count_detections(image_path, coco_class, cfg)
    verdict = _compute_verdict(detected_count, claimed_qty, ccfg["suspect_margin"])

    return {
        "verdict": verdict,
        "claimed_qty": claimed_qty,
        "detected_count": detected_count,
        "matched_class": coco_class,
        "mean_confidence": mean_conf,
        "message": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--order-item-id", required=True)
    parser.add_argument("--claimed-qty", type=int, required=True)
    args = parser.parse_args()

    result = verify_quantity(args.image, args.order_item_id, args.claimed_qty)
    print(f"order_item_id: {args.order_item_id}  claimed_qty: {args.claimed_qty}")
    if result["verdict"] == "unsupported":
        print(f"verdict: unsupported -- {result['message']}")
    else:
        print(
            f"verdict: {result['verdict']}  (matched_class={result['matched_class']!r}, "
            f"detected={result['detected_count']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
