"""
Generates the golden evaluation set: ~100 (query, relevant-products) pairs
used to measure Recall@K/NDCG@10 and gate every future model/prompt change
(design doc §7.2).

Query generation is template-based (deterministic, seeded) rather than
LLM-generated, since no LLM is wired up yet at this point in the build (that
lands Day 4 with Ollama). Once it is, `--llm ollama` can add paraphrased /
more naturalistic variants on top of these -- the template queries and
positive-set logic stay as the backbone either way.

Every query is grounded in a real sampled product ("seed"), stratified across
categories so the eval set isn't dominated by the catalog's largest category.
Positives = the seed product + other same-category products that also match
the attributes named in the query (same brand, same color) -- so Recall@K
isn't trivially "find this one exact item".

This file is a *draft* for human review, per the spec's "LLM + manual
spot-check" process: open data/eval/golden_set_review.csv, sanity-check a
sample of queries/positives, hand-edit data/eval/golden_set.jsonl directly if
anything looks wrong, then treat golden_set.jsonl as frozen ground truth.

Usage:
  python -m shoptalk.eval.generate_golden_set
  python -m shoptalk.eval.generate_golden_set --force   # overwrite existing golden set
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

from shoptalk.config import load_config


def readable_category(category: str) -> str:
    return category.replace("_", " ").lower()


def is_clean_attribute(value: str, max_len: int = 35) -> bool:
    """Rejects brand/color values unfit for a natural-language query: leaked
    internal codes (e.g. '#lightness'), values with embedded non-Latin script
    (mixed-language brand strings like 'AJC Collection(AJC コレクション)'),
    or implausibly long strings (run-on legal names)."""
    if not value or not value.strip():
        return False
    if len(value) > max_len:
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 .,'&\-]*$", value):
        return False
    return True


def price_threshold(price: float) -> int:
    """Round up to a plausible round-number price ceiling strictly above `price`."""
    for ceiling in (10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000):
        if price < ceiling:
            return ceiling
    return int(price) + 100


def make_query(row: pd.Series, rng: random.Random) -> tuple:
    """Returns (query_text, template_name, attrs_used)."""
    category = readable_category(row["category"]) if row["category"] else "product"
    brand = row["brand"] if is_clean_attribute(row["brand"]) else ""
    color = row["color"].lower() if is_clean_attribute(row["color"]) else ""
    price = row["price_usd"]

    templates = []
    if brand and color:
        templates.append(("brand_color", f"{color} {category} by {brand}", {"brand": brand, "color": color}))
    if brand:
        templates.append(("brand", f"{brand} {category}", {"brand": brand}))
    if color:
        templates.append(("color", f"{color} {category}", {"color": color}))
    templates.append(("price", f"{category} under ${price_threshold(price)}", {"price_ceiling": price_threshold(price)}))
    templates.append(("plain", f"{category}", {}))
    if brand and color:
        templates.append(
            ("affordable_brand_color", f"affordable {color} {category} from {brand}", {"brand": brand, "color": color})
        )

    name, query, attrs = rng.choice(templates)
    return query, name, attrs


def find_positives(seed_row: pd.Series, attrs: dict, df: pd.DataFrame, max_positives: int = 8) -> list:
    """Seed item + same-category products that also match the query's named attributes."""
    same_category = df[df["category"] == seed_row["category"]]

    mask = pd.Series(True, index=same_category.index)
    if "brand" in attrs:
        mask &= same_category["brand"] == seed_row["brand"]
    if "color" in attrs:
        mask &= same_category["color"].str.lower() == seed_row["color"].lower()
    if "price_ceiling" in attrs:
        mask &= same_category["price_usd"] < attrs["price_ceiling"]

    matches = same_category[mask]["item_id"].tolist()
    if seed_row["item_id"] not in matches:
        matches = [seed_row["item_id"]] + matches

    # de-dup, cap, keep seed first
    seen = set()
    positives = []
    for item_id in matches:
        if item_id not in seen:
            seen.add(item_id)
            positives.append(item_id)
        if len(positives) >= max_positives:
            break
    return positives


def stratified_sample(df: pd.DataFrame, n: int, rng_seed: int) -> pd.DataFrame:
    """Sample n products, capping how much any single category can contribute
    so the golden set mirrors the catalog's diversity, not just its head category."""
    n_categories = df["category"].nunique()
    per_category_cap = max(2, (n // max(n_categories, 1)) + 3)

    sampled = (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_category_cap), random_state=rng_seed), include_groups=False)
    )
    sampled["category"] = df.loc[sampled.index, "category"]
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=rng_seed)
    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing golden set")
    parser.add_argument("--size", type=int, default=None, help="override eval.golden_set_size")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, ecfg = cfg["data"], cfg["eval"]
    golden_path = Path(ecfg["golden_set_path"])

    if golden_path.exists() and not args.force:
        print(f"{golden_path} already exists -- pass --force to regenerate (this discards manual edits)")
        return 1

    df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")
    n = args.size or ecfg["golden_set_size"]
    rng = random.Random(ecfg["seed"])

    seeds = stratified_sample(df, n, ecfg["seed"])

    rows = []
    for i, (_, seed_row) in enumerate(seeds.iterrows()):
        query, template, attrs = make_query(seed_row, rng)
        positives = find_positives(seed_row, attrs, df)
        rows.append(
            {
                "query_id": f"q{i:04d}",
                "query": query,
                "template": template,
                "category": seed_row["category"],
                "seed_item_id": seed_row["item_id"],
                "positive_ids": positives,
            }
        )

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    review_csv = golden_path.parent / "golden_set_review.csv"
    review_df = pd.DataFrame(rows)
    review_df["positive_ids"] = review_df["positive_ids"].map(lambda ids: ", ".join(ids))
    review_df["num_positives"] = review_df["positive_ids"].str.count(",") + 1
    review_df.to_csv(review_csv, index=False)

    print(f"wrote {len(rows)} queries -> {golden_path}")
    print(f"wrote human-review CSV -> {review_csv}")
    print(f"avg positives/query: {review_df['num_positives'].mean():.1f}")
    print(
        "\nNext: spot-check golden_set_review.csv, hand-edit golden_set.jsonl for any "
        "wrong/nonsensical queries or positive sets, then treat it as frozen ground truth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
