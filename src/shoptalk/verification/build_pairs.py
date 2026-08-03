"""
Builds the Siamese verification pair dataset (design doc §2, §3.3b):
  positives = two different catalog photos of the SAME product (ABO's
              main_image_id + other_image_id fields -- multi-view images)
  negatives = hard negatives: the nearest CLIP neighbor from a DIFFERENT
              product in the SAME category (visually similar, genuinely
              confusable -- not a random unrelated product)

Downloads the extra "other view" images on demand (only for products used
here), reusing download_abo.py's image-fetch helpers rather than bloating the
main Day-1 download.

Input:  data/raw/listings/selected_products.jsonl (has other_image_id)
        data/processed/products.parquet
        data/chroma/ image collection (embed_image.py, for hard-neg mining)
Output: data/verification/pairs.jsonl

Usage:
  python -m shoptalk.verification.build_pairs
  python -m shoptalk.verification.build_pairs --limit 100   # smoke test
"""
import argparse
import json
import random
import sys
from pathlib import Path

import chromadb
import pandas as pd

from shoptalk.config import load_config
from shoptalk.data.download_abo import download_images, load_image_index


def load_multiview_products(manifest_path: Path) -> list:
    """Products with at least one other_image_id besides the main image."""
    products = []
    with open(manifest_path) as f:
        for line in f:
            obj = json.loads(line)
            others = obj.get("other_image_id") or []
            if obj.get("main_image_id") and others:
                products.append(
                    {
                        "item_id": obj["item_id"],
                        "main_image_id": obj["main_image_id"],
                        "other_image_id": others[0],
                    }
                )
    return products


def mine_hard_negative(item_id: str, category: str, image_collection, products_df: pd.DataFrame, rng: random.Random):
    """Nearest CLIP neighbor in the same category, excluding the product itself.
    Falls back to a random same-category product if the ANN query comes up empty."""
    row = products_df.loc[products_df["item_id"] == item_id]
    if row.empty:
        return None
    try:
        result = image_collection.get(ids=[item_id], include=["embeddings"])
        if len(result["ids"]) == 0:
            raise ValueError("not indexed")
        query_embedding = result["embeddings"][0]
        neighbors = image_collection.query(
            query_embeddings=[query_embedding], n_results=10,
            where={"category": category}, include=["metadatas"],
        )
        for neighbor_id, meta in zip(neighbors["ids"][0], neighbors["metadatas"][0]):
            if neighbor_id != item_id:
                return {"item_id": neighbor_id, "image_path": meta["image_path"]}
    except Exception:
        pass

    same_cat = products_df[
        (products_df["category"] == category)
        & (products_df["item_id"] != item_id)
        & (products_df["image_available"])
    ]
    if same_cat.empty:
        return None
    negative = same_cat.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
    return {"item_id": negative["item_id"], "image_path": negative["image_path"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="max products used to build pairs")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, clcfg = cfg["data"], cfg["clip"]
    raw_dir = Path(dcfg["raw_dir"])
    rng = random.Random(dcfg["seed"])

    manifest_path = raw_dir / "listings" / "selected_products.jsonl"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found -- run download_abo.py first", file=sys.stderr)
        return 1

    products_df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")
    multiview = load_multiview_products(manifest_path)
    multiview = [p for p in multiview if p["item_id"] in set(products_df["item_id"])]
    if args.limit:
        multiview = multiview[: args.limit]
    print(f"{len(multiview)} products have a second view available")

    print("downloading second-view images...")
    image_index = load_image_index(raw_dir / "images.csv.gz")
    other_ids = [p["other_image_id"] for p in multiview]
    download_images(dcfg["base_url"], dcfg["images_prefix"], other_ids, image_index, raw_dir / "images")

    client = chromadb.PersistentClient(path=clcfg["chroma_dir"])
    image_collection = client.get_collection(clcfg["collection_name"])

    image_path_by_id = products_df.set_index("item_id")["image_path"]
    category_by_id = products_df.set_index("item_id")["category"]

    pairs = []
    for p in multiview:
        item_id = p["item_id"]
        main_path = image_path_by_id.get(item_id)
        other_path = raw_dir / "images" / f"{p['other_image_id']}.jpg"
        if not main_path or not other_path.exists():
            continue

        pairs.append(
            {
                "pair_id": f"{item_id}_pos",
                "image_a_path": main_path,
                "image_b_path": str(other_path),
                "item_id_a": item_id,
                "item_id_b": item_id,
                "label": 1,
            }
        )

        negative = mine_hard_negative(item_id, category_by_id[item_id], image_collection, products_df, rng)
        if negative:
            pairs.append(
                {
                    "pair_id": f"{item_id}_neg",
                    "image_a_path": main_path,
                    "image_b_path": negative["image_path"],
                    "item_id_a": item_id,
                    "item_id_b": negative["item_id"],
                    "label": 0,
                }
            )

    out_path = Path(cfg["verification"]["pairs_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(pair) + "\n" for pair in pairs)

    n_pos = sum(1 for p in pairs if p["label"] == 1)
    print(f"wrote {len(pairs)} pairs ({n_pos} positive, {len(pairs) - n_pos} negative) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
