"""
Evidently drift report: reference query distribution (the golden eval set)
vs a simulated shifted window (design doc §6: "distribution of query
embeddings (PSI/KS on projected dims), query length... vs reference
window"). The shifted set is deliberately built to differ (out-of-catalog
vocabulary, unusual lengths) rather than resampled from the same
distribution, so the report demonstrates the pipeline actually *detects*
drift when it's present, not just that it runs.

Usage:
  python -m shoptalk.monitoring.drift_report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from shoptalk.config import load_config

SIMULATED_SHIFTED_QUERIES = [
    "3D printer filament spool 1.75mm",
    "vintage vinyl record player with bluetooth",
    "organic compost bin for backyard gardening",
    "telescope for viewing planets at night",
    "industrial label maker with thermal printing",
    "aquarium co2 diffuser for planted tank",
    "drone propeller replacement kit carbon fiber",
    "sous vide immersion circulator precision cooker",
    "asdkj qwoeiru zxcvbn product query gibberish test",
    "a",
    "the best most amazing incredible super premium ultra deluxe "
    "professional grade item that money can buy today right now immediately",
]


def build_feature_frame(queries: list, model: SentenceTransformer, pca: PCA) -> pd.DataFrame:
    embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    # see eval_finetune.py: macOS Accelerate BLAS emits spurious matmul
    # RuntimeWarnings here with no actual NaN/Inf in the output.
    with np.errstate(all="ignore"):
        projected = pca.transform(embeddings)
    df = pd.DataFrame(projected, columns=[f"embedding_pc{i}" for i in range(projected.shape[1])])
    df["query_length_chars"] = [len(q) for q in queries]
    df["query_length_words"] = [len(q.split()) for q in queries]
    return df


def main():
    cfg = load_config()
    golden_path = Path(cfg["eval"]["golden_set_path"])
    if not golden_path.exists():
        print(f"error: {golden_path} not found -- run generate_golden_set.py first", file=sys.stderr)
        return 1

    reference_queries = [json.loads(line)["query"] for line in open(golden_path)]
    print(f"reference window: {len(reference_queries)} queries (golden eval set)")
    print(f"current window: {len(SIMULATED_SHIFTED_QUERIES)} simulated shifted queries")

    model = SentenceTransformer(cfg["embeddings"]["text_model"])
    ref_embeddings = model.encode(reference_queries, normalize_embeddings=True, show_progress_bar=False)

    n_components = min(5, len(reference_queries) - 1, ref_embeddings.shape[1])
    with np.errstate(all="ignore"):
        pca = PCA(n_components=n_components, random_state=cfg["data"]["seed"]).fit(ref_embeddings)

    reference_df = build_feature_frame(reference_queries, model, pca)
    current_df = build_feature_frame(SIMULATED_SHIFTED_QUERIES, model, pca)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    html_path = results_dir / "day6_drift_report.html"
    report.save_html(str(html_path))

    drift_summary = report.as_dict()["metrics"][0]["result"]
    print(f"\ndataset drift detected: {drift_summary['dataset_drift']}")
    print(f"drifted columns: {drift_summary['number_of_drifted_columns']}/{drift_summary['number_of_columns']}")
    print(f"wrote {html_path}")

    with open(results_dir / "day6_drift_report.md", "w") as f:
        f.write("# Day 6 — Data drift report\n\n")
        f.write(f"- Reference window: {len(reference_queries)} queries (golden eval set)\n")
        f.write(
            f"- Current window: {len(SIMULATED_SHIFTED_QUERIES)} simulated shifted queries "
            "(deliberately out-of-catalog vocabulary + unusual lengths)\n"
        )
        f.write(f"- Dataset drift detected: **{drift_summary['dataset_drift']}**\n")
        f.write(
            f"- Drifted columns: {drift_summary['number_of_drifted_columns']}"
            f"/{drift_summary['number_of_columns']}\n"
        )
        f.write("- Full interactive report: `day6_drift_report.html`\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
