"""
Downloads a subset of the Amazon Berkeley Objects (ABO) dataset directly from
its public S3 bucket (no AWS credentials needed -- files are individually
addressable, so we avoid pulling the full multi-GB tar archives).

Produces, under data/raw/:
  listings/listings_*.json.gz   - raw listing shards (as published)
  images.csv.gz                 - image_id -> path lookup (as published)
  images/<image_id>.jpg         - only the main-image for each selected,
                                   English-language product

Usage:
  python -m shoptalk.data.download_abo
  python -m shoptalk.data.download_abo --limit 200 --skip-images   # smoke test
"""
import argparse
import csv
import gzip
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from shoptalk.config import load_config

TIMEOUT = 30
MAX_WORKERS = 16


def _get(url: str) -> bytes:
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def download_file(base_url: str, remote_path: str, dest: Path, desc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {desc} already present at {dest}")
        return dest
    print(f"  [get]  {desc} <- {base_url}/{remote_path}")
    content = _get(f"{base_url}/{remote_path}")
    dest.write_bytes(content)
    return dest


def load_english_listings(shard_paths: list[Path], language_tag: str) -> list[dict]:
    """Parse listing shards, keep only products with an item_name in the target language."""
    products = []
    for shard_path in shard_paths:
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                item_names = obj.get("item_name", [])
                if any(n.get("language_tag", "").startswith(language_tag[:2]) for n in item_names):
                    products.append(obj)
    return products


def load_image_index(images_csv_path: Path) -> dict:
    """image_id -> relative path (e.g. '14/14fe8812.jpg')."""
    index = {}
    with gzip.open(images_csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index[row["image_id"]] = row["path"]
    return index


def download_images(
    base_url: str,
    images_prefix: str,
    image_ids: list[str],
    image_index: dict,
    images_dir: Path,
) -> dict:
    """Downloads each image_id's small-variant jpg. Returns {image_id: local_path or None}."""
    images_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def _fetch_one(image_id: str):
        rel_path = image_index.get(image_id)
        if rel_path is None:
            return image_id, None
        dest = images_dir / f"{image_id}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            return image_id, str(dest)
        try:
            content = _get(f"{base_url}/{images_prefix}/{rel_path}")
            dest.write_bytes(content)
            return image_id, str(dest)
        except requests.RequestException:
            return image_id, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_fetch_one, iid) for iid in image_ids]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images"):
            image_id, path = fut.result()
            results[image_id] = path
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap product count (smoke tests)")
    parser.add_argument("--skip-images", action="store_true", help="skip downloading product images")
    args = parser.parse_args()

    cfg = load_config()
    dcfg = cfg["data"]
    base_url = dcfg["base_url"]
    raw_dir = Path(dcfg["raw_dir"])
    random.seed(dcfg["seed"])

    print("1/4 Downloading listing shards...")
    shard_paths = []
    for shard in dcfg["listings_shards"]:
        dest = raw_dir / "listings" / shard
        download_file(base_url, f"listings/metadata/{shard}", dest, shard)
        shard_paths.append(dest)

    print("2/4 Downloading image index (images.csv.gz)...")
    images_csv_path = download_file(
        base_url, dcfg["images_metadata"], raw_dir / "images.csv.gz", "images.csv.gz"
    )

    print("3/4 Filtering to English-language listings...")
    products = load_english_listings(shard_paths, dcfg["language_tag"])
    print(f"  {len(products)} English-language products found across shards")

    target = args.limit or dcfg["target_product_count"]
    if len(products) > target:
        products = random.sample(products, target)
    print(f"  sampled down to {len(products)} products")

    manifest_path = raw_dir / "listings" / "selected_products.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for p in products:
            f.write(json.dumps(p) + "\n")
    print(f"  wrote manifest -> {manifest_path}")

    if args.skip_images:
        print("4/4 Skipping image download (--skip-images)")
        return

    print("4/4 Downloading main-image for each selected product...")
    image_index = load_image_index(images_csv_path)
    image_ids = [p["main_image_id"] for p in products if p.get("main_image_id")]
    results = download_images(
        base_url, dcfg["images_prefix"], image_ids, image_index, raw_dir / "images"
    )
    missing = sum(1 for v in results.values() if v is None)
    print(f"  downloaded {len(results) - missing}/{len(results)} images ({missing} missing)")


if __name__ == "__main__":
    sys.exit(main())
