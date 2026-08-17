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
    """Recall@k (fraction of THIS query's positives found in the top k) and
    Precision@k (fraction of the top k that are positives -- the submission
    guideline's "percent of relevant results (True positives) from the
    retrieved results"). Precision@k reads much lower than Recall@k here by
    construction: most golden-set queries have only 1-2 labeled positives
    (capping Precision@10 at 0.1-0.2 even for a perfect retrieval), while a
    handful have up to 8 -- see the golden set's own positives-per-query
    distribution, not a sign retrieval is worse than Recall@k suggests."""
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
    precisions = {k: [] for k in k_values}
    for i, row in enumerate(golden_set):
        positives = set(row["positive_ids"]) & set(item_ids)
        if not positives:
            continue
        ranked_idx = np.argsort(-sims[i])
        for k in k_values:
            top_k_ids = {item_ids[j] for j in ranked_idx[:k]}
            hits = len(top_k_ids & positives)
            recalls[k].append(hits / len(positives))
            precisions[k].append(hits / k)
    metrics = {f"recall@{k}": (float(np.mean(v)) if v else 0.0) for k, v in recalls.items()}
    metrics.update({f"precision@{k}": (float(np.mean(v)) if v else 0.0) for k, v in precisions.items()})
    return metrics


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

    # Always the pristine pretrained model -- see config.yaml's base_text_model
    # comment for why this can't just be cfg["embeddings"]["text_model"] (that
    # may itself already be a fine-tuned checkpoint once one is promoted).
    base_text_model = cfg["embeddings"].get("base_text_model", cfg["embeddings"]["text_model"])
    print(f"evaluating base (pretrained) model ({base_text_model})...")
    base_model = SentenceTransformer(base_text_model)
    base_metrics = recall_at_k(base_model, products_df, golden_set, k_values)

    print("evaluating fine-tuned model...")
    ft_model = SentenceTransformer(args.finetuned_path)
    ft_metrics = recall_at_k(ft_model, products_df, golden_set, k_values)

    metric_names = [f"recall@{k}" for k in k_values] + [f"precision@{k}" for k in k_values]
    table = pd.DataFrame(
        {
            "metric": metric_names,
            "base_model": [base_metrics[m] for m in metric_names],
            "finetuned_model": [ft_metrics[m] for m in metric_names],
        }
    )
    table["uplift"] = table["finetuned_model"] - table["base_model"]
    print("\n=== Fine-tune uplift ===")
    print(table.to_string(index=False))
    print(
        "\nNote: Precision@k is much lower than Recall@k on this golden set by "
        "construction -- most queries have only 1-2 labeled positives, capping "
        "Precision@10 well below 1.0 even for perfect retrieval."
    )

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "day5_finetune_eval.md"
    with open(md_path, "w") as f:
        f.write("# Day 5 — Embedding fine-tune evaluation\n\n")
        f.write(f"Golden set: {len(golden_set)} queries\n\n")
        f.write(table.to_markdown(index=False))
        f.write(
            "\n\n> Precision@k is much lower than Recall@k here by construction, "
            "not because retrieval is weaker than Recall@k suggests -- most "
            "golden-set queries have only 1-2 labeled positives, which caps "
            "Precision@10 at 0.1-0.2 even for perfect retrieval; a handful of "
            "queries with up to 8 positives pull the average up from there.\n"
        )
    print(f"\nwrote {md_path}")

    # Machine-readable twin of the markdown table -- so a caller (e.g. the
    # Airflow retrain DAG, which runs this as a subprocess rather than
    # importing it -- see that DAG's own notes on why) can read the exact
    # numbers back without parsing markdown.
    json_path = results_dir / "day5_finetune_eval.json"
    json_path.write_text(json.dumps({"base_model": base_metrics, "finetuned_model": ft_metrics}))
    print(f"wrote {json_path}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    for name, metrics, model_ref in [
        ("base_embedding_model", base_metrics, base_text_model),
        ("finetuned_embedding_model", ft_metrics, args.finetuned_path),
    ]:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", model_ref)
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
    print("logged both runs to MLflow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
