"""
Simulates user-item interaction history (design doc §3.2c) for
personalization: each synthetic user has 1-2 preferred categories and
"clicks"/"purchases" a sequence of products weighted toward those
categories (with some noise, so profiles aren't perfectly clean) -- a
plausible-enough taste signal to demonstrate reranking uplift without real
user data (which doesn't exist for this project).

Usage:
  python -m shoptalk.personalization.simulate_interactions
"""
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

from shoptalk.config import load_config


def main():
    cfg = load_config()
    pcfg = cfg["personalization"]
    rng = random.Random(cfg["data"]["seed"])

    products_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    categories = products_df["category"].unique().tolist()
    if len(categories) < 2:
        print("error: need at least 2 categories in the catalog to simulate meaningful taste profiles", file=sys.stderr)
        return 1

    interactions = []
    now = time.time()
    for u in range(pcfg["num_synthetic_users"]):
        user_id = f"user_{u:04d}"
        preferred = rng.sample(categories, k=min(2, len(categories)))
        n_interactions = rng.randint(5, 15)

        for i in range(n_interactions):
            # 80% of interactions from preferred categories, 20% exploratory noise
            if rng.random() < 0.8:
                category = rng.choice(preferred)
            else:
                category = rng.choice(categories)

            candidates = products_df[products_df["category"] == category]
            if candidates.empty:
                continue
            item = candidates.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]

            interactions.append(
                {
                    "user_id": user_id,
                    "item_id": item["item_id"],
                    "category": category,
                    "interaction_type": rng.choice(["click", "click", "click", "purchase"]),
                    # older interactions further in the past, for decay weighting
                    "timestamp": now - (n_interactions - i) * 3600,
                }
            )

    out_path = Path(pcfg["interactions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in interactions:
            f.write(json.dumps(row) + "\n")

    print(f"simulated {len(interactions)} interactions across {pcfg['num_synthetic_users']} users -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
