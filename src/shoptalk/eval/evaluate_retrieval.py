"""
Runs the golden eval set through stage-1-only retrieval and full two-stage
(stage-1 + cross-encoder rerank), computes Recall@10/50/100, MRR, and NDCG@10
via ranx for both, and logs an uplift comparison to MLflow.

Both runs reuse the SAME stage-1 top-100 candidate pool per query (the
reranker only reorders it) -- so Recall@100 should come out ~identical
between the two runs; that's expected and confirms the reranker isn't
silently changing the candidate set, only its ordering. The uplift shows up
in Recall@10/50, MRR, and NDCG@10.

Usage:
  python -m shoptalk.eval.evaluate_retrieval
  python -m shoptalk.eval.evaluate_retrieval --limit 20   # smoke test on first 20 queries
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

from shoptalk.config import load_config
from shoptalk.retrieval.rerank import rerank
from shoptalk.retrieval.search import search as stage1_search

METRICS = ["recall@10", "recall@50", "recall@100", "mrr", "ndcg@10"]


def load_golden_set(path: Path, limit: int = None) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def run_to_ranx(query_id: str, hits: list, score_key: str) -> dict:
    return {hit["item_id"]: float(hit[score_key]) for hit in hits}


def mlflow_safe_metrics(metrics: dict) -> dict:
    """MLflow metric names may only contain alphanumerics/_/-/./space/:// -- ranx
    uses '@' (e.g. 'recall@10'), so remap to 'recall_at_10' for logging only."""
    return {k.replace("@", "_at_"): v for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap queries evaluated (smoke tests)")
    args = parser.parse_args()

    cfg = load_config()
    ecfg, rcfg, mcfg = cfg["eval"], cfg["retrieval"], cfg["mlflow"]

    golden_set_path = Path(ecfg["golden_set_path"])
    if not golden_set_path.exists():
        print(f"error: {golden_set_path} not found -- run generate_golden_set.py first", file=sys.stderr)
        return 1

    queries = load_golden_set(golden_set_path, args.limit)
    print(f"evaluating {len(queries)} queries from {golden_set_path}")

    qrels_dict, baseline_run_dict, two_stage_run_dict = {}, {}, {}

    for q in tqdm(queries, desc="retrieving"):
        qid = q["query_id"]
        qrels_dict[qid] = {item_id: 1 for item_id in q["positive_ids"]}

        stage1_hits = stage1_search(q["query"], top_k=rcfg["stage1_k"], cfg=cfg)
        baseline_run_dict[qid] = run_to_ranx(qid, stage1_hits, "score")

        two_stage_hits = rerank(q["query"], stage1_hits, top_k=rcfg["stage1_k"], cfg=cfg)
        two_stage_run_dict[qid] = run_to_ranx(qid, two_stage_hits, "rerank_score")

    qrels = Qrels(qrels_dict)
    baseline_metrics = evaluate(qrels, Run(baseline_run_dict), METRICS)
    two_stage_metrics = evaluate(qrels, Run(two_stage_run_dict), METRICS)

    table = pd.DataFrame(
        {
            "metric": METRICS,
            "stage1_baseline": [baseline_metrics[m] for m in METRICS],
            "two_stage_reranked": [two_stage_metrics[m] for m in METRICS],
        }
    )
    table["uplift"] = table["two_stage_reranked"] - table["stage1_baseline"]
    table["uplift_pct"] = (table["uplift"] / table["stage1_baseline"].replace(0, pd.NA) * 100).round(1)

    print("\n=== Two-stage retrieval uplift ===")
    print(table.to_string(index=False))

    results_dir = Path(ecfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "day2_retrieval_eval.md"
    with open(md_path, "w") as f:
        f.write("# Day 2 — Two-stage retrieval evaluation\n\n")
        f.write(f"Golden set: {len(queries)} queries ({golden_set_path})\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")
    print(f"\nwrote {md_path}")

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])

    with mlflow.start_run(run_name="stage1_baseline"):
        mlflow.log_param("text_model", cfg["embeddings"]["text_model"])
        mlflow.log_param("stage1_k", rcfg["stage1_k"])
        mlflow.log_param("golden_set_size", len(queries))
        mlflow.log_metrics(mlflow_safe_metrics(baseline_metrics))

    with mlflow.start_run(run_name="two_stage_reranked"):
        mlflow.log_param("text_model", cfg["embeddings"]["text_model"])
        mlflow.log_param("rerank_model", rcfg["rerank_model"])
        mlflow.log_param("stage1_k", rcfg["stage1_k"])
        mlflow.log_param("golden_set_size", len(queries))
        mlflow.log_metrics(mlflow_safe_metrics(two_stage_metrics))
        mlflow.log_artifact(str(md_path))

    print(f"logged both runs to MLflow ({mcfg['tracking_uri']}, experiment={mcfg['experiment_name']!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
