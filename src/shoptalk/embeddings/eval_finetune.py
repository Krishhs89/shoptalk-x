"""
Compares the pretrained vs fine-tuned embedding model on the golden eval set
(problem statement Deliverables: "finetuned models can be compared with
pretrained models"). Brute-force cosine top-K over the full catalog -- no ANN
needed at this scale -- reporting Recall@10/50 pre vs post fine-tuning.

Usage:
  python -m shoptalk.embeddings.eval_finetune --finetuned-path data/models/bge-finetuned
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from shoptalk.config import load_config
from shoptalk.retrieval.search import BGE_QUERY_INSTRUCTION


def recall_at_k(model: SentenceTransformer, products_df: pd.DataFrame, golden_set: list, k_values: list) -> dict:
    doc_embeddings = model.encode(
        products_df["document"].tolist(), normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    item_ids = products_df["item_id"].tolist()

    queries = [BGE_QUERY_INSTRUCTION + row["query"] for row in golden_set]
    query_embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

    # macOS Accelerate BLAS emits spurious divide-by-zero/overflow RuntimeWarnings
    # on this matmul with no actual NaN/Inf in the output (verified) -- known
    # Accelerate quirk, not a real numerical issue; suppressed to avoid alarming
    # users on Apple Silicon.
    with np.errstate(all="ignore"):
        sims = query_embeddings @ doc_embeddings.T
    recalls = {k: [] for k in k_values}
    for i, row in enumerate(golden_set):
        positives = set(row["positive_ids"]) & set(item_ids)
        if not positives:
            continue
        ranked_idx = np.argsort(-sims[i])
        for k in k_values:
            top_k_ids = {item_ids[j] for j in ranked_idx[:k]}
            recalls[k].append(len(top_k_ids & positives) / len(positives))
    return {f"recall@{k}": (float(np.mean(v)) if v else 0.0) for k, v in recalls.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-path", required=True)
    args = parser.parse_args()

    cfg = load_config()
    products_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    golden_path = Path(cfg["eval"]["golden_set_path"])
    if not golden_path.exists():
        print(f"error: {golden_path} not found -- run generate_golden_set.py first", file=sys.stderr)
        return 1
    golden_set = [json.loads(line) for line in open(golden_path)]
    k_values = [10, 50]

    print("evaluating base (pretrained) model...")
    base_model = SentenceTransformer(cfg["embeddings"]["text_model"])
    base_metrics = recall_at_k(base_model, products_df, golden_set, k_values)

    print("evaluating fine-tuned model...")
    ft_model = SentenceTransformer(args.finetuned_path)
    ft_metrics = recall_at_k(ft_model, products_df, golden_set, k_values)

    table = pd.DataFrame(
        {
            "metric": [f"recall@{k}" for k in k_values],
            "base_model": [base_metrics[f"recall@{k}"] for k in k_values],
            "finetuned_model": [ft_metrics[f"recall@{k}"] for k in k_values],
        }
    )
    table["uplift"] = table["finetuned_model"] - table["base_model"]
    print("\n=== Fine-tune uplift ===")
    print(table.to_string(index=False))

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "day5_finetune_eval.md"
    with open(md_path, "w") as f:
        f.write("# Day 5 — Embedding fine-tune evaluation\n\n")
        f.write(f"Golden set: {len(golden_set)} queries\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")
    print(f"\nwrote {md_path}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    for name, metrics, model_ref in [
        ("base_embedding_model", base_metrics, cfg["embeddings"]["text_model"]),
        ("finetuned_embedding_model", ft_metrics, args.finetuned_path),
    ]:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", model_ref)
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
    print("logged both runs to MLflow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
