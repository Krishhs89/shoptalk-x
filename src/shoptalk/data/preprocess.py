"""
Cleans and joins the raw ABO listing fields into a flat product table used by
every downstream module (embeddings, retrieval, RAG, UI).

Input:  data/raw/listings/selected_products.jsonl  (written by download_abo.py)
        data/raw/images/<image_id>.jpg
Output: data/processed/products.parquet + products.csv

IMPORTANT: ABO has no price field. The problem-statement example query
("red shirt under $50") implies price filtering, so we attach a SYNTHETIC
price (deterministic per item_id, category-aware range) and mark it with
price_is_synthetic=True. Never present this as real Amazon pricing.

Usage:
  python -m shoptalk.data.preprocess
  python -m shoptalk.data.preprocess --limit 200   # smoke test
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from shoptalk.config import load_config

# Rough per-category price bands (USD) used only to make the synthetic price
# plausible relative to the product type -- NOT derived from real Amazon data.
CATEGORY_PRICE_BANDS = {
    "SHOES": (25, 150),
    "SOFA": (300, 2000),
    "CHAIR": (60, 500),
    "TABLE": (80, 800),
    "LAMP": (20, 200),
    "RUG": (30, 400),
    "BED": (200, 1500),
    "MATTRESS": (150, 1200),
    "CELLULAR_PHONE_CASE": (5, 40),
    "HEADPHONES": (15, 300),
    "BACKPACK": (20, 150),
    "HANDBAG": (25, 250),
    "WATCH": (20, 400),
    "JEWELRY": (10, 300),
    "OUTDOOR_LIVING": (30, 600),
    "HOME_BED_AND_BATH": (10, 150),
    "KITCHEN": (10, 200),
}
DEFAULT_PRICE_BAND = (10, 200)

WHITESPACE_RE = re.compile(r"\s+")


def _first_english(field_list: list, language_prefix: str = "en") -> str:
    """Given ABO's [{language_tag, value}, ...] lists, return the first English value."""
    if not field_list:
        return ""
    for entry in field_list:
        if entry.get("language_tag", "").startswith(language_prefix):
            return str(entry.get("value", ""))
    # fall back to first entry if no English tag matched (shouldn't happen post-filter)
    return str(field_list[0].get("value", ""))


def _all_english(field_list: list, language_prefix: str = "en") -> list:
    """Collects English values, de-duplicated (case-insensitive, order-preserving).

    ABO listings often repeat the same bullet points / keywords across several
    English locale tags (en_US, en_GB, en_IN, ...), so a naive collection
    triples or quadruples the text without adding information.
    """
    seen = set()
    out = []
    for e in field_list or []:
        if not e.get("language_tag", "").startswith(language_prefix):
            continue
        value = str(e.get("value", "")).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def clean_text(text: str) -> str:
    text = text.replace("\u200b", " ")  # zero-width space seen in some ABO fields
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def synthetic_price(item_id: str, category: str) -> float:
    """Deterministic pseudo-random price seeded by item_id so re-runs are stable."""
    low, high = CATEGORY_PRICE_BANDS.get(category, DEFAULT_PRICE_BAND)
    digest = hashlib.sha256(item_id.encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF  # deterministic float in [0, 1)
    return round(low + frac * (high - low), 2)


def build_record(obj: dict, images_dir: Path) -> dict:
    item_id = obj["item_id"]
    item_name = _first_english(obj.get("item_name", []))
    bullet_points = _all_english(obj.get("bullet_point", []))
    keywords = _all_english(obj.get("item_keywords", []))
    brand = _first_english(obj.get("brand", []))
    color = _first_english(obj.get("color", []))
    style = _first_english(obj.get("style", []))
    category = obj.get("product_type", [{}])[0].get("value", "") if obj.get("product_type") else ""

    description = clean_text(" ".join(bullet_points))
    item_name = clean_text(item_name)
    keywords_str = clean_text(", ".join(keywords))

    document = clean_text(
        " | ".join(filter(None, [item_name, category, brand, color, style, description, keywords_str]))
    )

    image_id = obj.get("main_image_id")
    image_path = images_dir / f"{image_id}.jpg" if image_id else None
    image_available = bool(image_path and image_path.exists())

    return {
        "item_id": item_id,
        "item_name": item_name,
        "category": category,
        "brand": brand,
        "color": color,
        "style": style,
        "description": description,
        "keywords": keywords_str,
        "document": document,
        "image_id": image_id,
        "image_path": str(image_path) if image_available else None,
        "image_available": image_available,
        "price_usd": synthetic_price(item_id, category),
        "price_is_synthetic": True,
        "country": obj.get("country", ""),
        "domain_name": obj.get("domain_name", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap rows processed (smoke tests)")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "listings" / "selected_products.jsonl"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found -- run download_abo.py first", file=sys.stderr)
        return 1

    images_dir = raw_dir / "images"
    records = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            obj = json.loads(line)
            records.append(build_record(obj, images_dir))

    df = pd.DataFrame(records)
    # drop any row that ended up with no usable name/description after cleaning
    before = len(df)
    df = df[(df["item_name"] != "") | (df["description"] != "")].reset_index(drop=True)
    dropped = before - len(df)

    out_parquet = processed_dir / "products.parquet"
    out_csv = processed_dir / "products.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"processed {len(df)} products ({dropped} dropped for empty name+description)")
    print(f"  images available: {df['image_available'].sum()} / {len(df)}")
    print(f"  categories: {df['category'].nunique()}")
    print(f"  wrote {out_parquet}")
    print(f"  wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
