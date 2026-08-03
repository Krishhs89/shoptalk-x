"""
Evaluates personalization uplift (design doc §3.2c: "Evaluate uplift in
Precision@K for repeat users vs non-personalized baseline"):
for each synthetic user, hold out their single most recent interaction,
build the profile from the rest, then check whether a generic
same-category query surfaces the held-out item in the top-K -- with vs
without the personalization blend.

Usage:
  python -m shoptalk.personalization.evaluate_personalization
"""
import sys
from pathlib import Path

import mlflow
import pandas as pd

from shoptalk.config import load_config
from shoptalk.eval.generate_golden_set import readable_category
from shoptalk.personalization.rerank_blend import personalized_rerank
from shoptalk.retrieval.two_stage import two_stage_search


def main():
    cfg = load_config()
    pcfg = cfg["personalization"]

    interactions_path = Path(pcfg["interactions_path"])
    if not interactions_path.exists():
        print(f"error: {interactions_path} not found -- run simulate_interactions.py first", file=sys.stderr)
        return 1

    interactions_df = pd.read_json(interactions_path, lines=True)
    top_k = cfg["retrieval"]["top_k"]

    baseline_hits, personalized_hits, n_evaluated = 0, 0, 0

    for user_id, group in interactions_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue  # need some history left after holding one out
        held_out = group.iloc[-1]
        query = readable_category(held_out["category"])

        baseline = two_stage_search(query, top_k=top_k, cfg=cfg)
        baseline_ids = {h["item_id"] for h in baseline}

        personalized = personalized_rerank(query, user_id, top_k=top_k, cfg=cfg)
        personalized_ids = {h["item_id"] for h in personalized}

        n_evaluated += 1
        baseline_hits += int(held_out["item_id"] in baseline_ids)
        personalized_hits += int(held_out["item_id"] in personalized_ids)

    if n_evaluated == 0:
        print("no users had enough interaction history to evaluate", file=sys.stderr)
        return 1

    baseline_precision = baseline_hits / n_evaluated
    personalized_precision = personalized_hits / n_evaluated

    print(f"evaluated {n_evaluated} users (held-out most-recent interaction each)")
    print(f"baseline hit-rate@{top_k}:      {baseline_precision:.3f}")
    print(f"personalized hit-rate@{top_k}:  {personalized_precision:.3f}")
    print(f"uplift: {personalized_precision - baseline_precision:+.3f}")

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stretch_personalization_eval.md", "w") as f:
        f.write("# Stretch — Personalization rerank uplift\n\n")
        f.write(f"- Users evaluated: {n_evaluated}\n")
        f.write(f"- Baseline hit-rate@{top_k}: {baseline_precision:.3f}\n")
        f.write(f"- Personalized hit-rate@{top_k}: {personalized_precision:.3f}\n")
        f.write(f"- Uplift: {personalized_precision - baseline_precision:+.3f}\n")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="personalization_uplift"):
        mlflow.log_params({"alpha": pcfg["alpha"], "beta": pcfg["beta"], "decay": pcfg["decay"]})
        mlflow.log_metrics(
            {
                f"hit_rate_at_{top_k}_baseline": baseline_precision,
                f"hit_rate_at_{top_k}_personalized": personalized_precision,
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
