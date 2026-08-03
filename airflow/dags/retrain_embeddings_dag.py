"""
Drift/scheduled retraining DAG (design doc §7.3): pull data -> rebuild
hard-negative triplets -> fine-tune embeddings -> evaluate vs the golden
set -> promote only if it beats the current model -> (placeholder) trigger
CI/CD deploy of the new artifact.

Simplified per the execution plan's Day-6 scope decision: one linear DAG,
demoed locally via `docker compose -f docker-compose.airflow.yml up`, not a
multi-DAG production Airflow deployment. Each task shells out to the same
scripts you'd run by hand (finetune_text.py, eval_finetune.py) so the DAG
and the manual workflow can never drift apart.

Trigger: scheduled (weekly) OR manually via the Airflow UI/CLI when a drift
alert fires (design doc §7.3, §6 "Alerting"). No automatic drift->DAG wiring
in this repo -- that's a `TriggerDagRunOperator` call from wherever
drift_report.py's "dataset drift detected" result is checked (e.g. a cron
job or the monitoring stack), documented here rather than wired live since
we don't have a real alerting pipeline (PagerDuty/Slack) to wire it to yet.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "shoptalk-x",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_pull_data(**context):
    """In production: re-run download_abo.py for newly listed products and
    incrementally upsert (design doc §7.4). For this demo DAG, the catalog
    is assumed already present under data/processed/ -- this task just
    validates it exists so failures surface early and clearly."""
    from pathlib import Path

    from shoptalk.config import load_config

    cfg = load_config()
    products_path = Path(cfg["data"]["processed_dir"]) / "products.parquet"
    if not products_path.exists():
        raise FileNotFoundError(
            f"{products_path} missing -- run download_abo.py + preprocess.py before this DAG"
        )
    print(f"catalog present: {products_path}")


def task_rebuild_golden_set(**context):
    """Regenerates the golden set only if it doesn't already exist -- an
    existing hand-reviewed golden_set.jsonl is treated as frozen ground
    truth (see generate_golden_set.py's own --force guard) and never
    silently overwritten by a scheduled run."""
    import sys

    from shoptalk.eval.generate_golden_set import main as generate_golden_set_main

    sys.argv = ["generate_golden_set.py"]  # avoid argparse seeing Airflow's own CLI args
    generate_golden_set_main()


def task_finetune_embeddings(**context):
    import sys

    from shoptalk.embeddings.finetune_text import main as finetune_main

    sys.argv = ["finetune_text.py"]  # use config.yaml defaults for epochs/batch-size
    finetune_main()


def task_evaluate_and_promote(**context):
    """Compares the freshly fine-tuned model against the currently-served
    base model on the golden set; only registers/promotes if Recall@10
    actually improves -- the CI-style regression gate from design doc §7.2
    ("every model/prompt change evaluated against it before promotion")."""
    import json
    from pathlib import Path

    import mlflow
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from shoptalk.config import load_config
    from shoptalk.embeddings.eval_finetune import recall_at_k
    from shoptalk.retrieval.search import BGE_QUERY_INSTRUCTION  # noqa: F401 (documents the query convention)

    cfg = load_config()
    products_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    golden_set = [json.loads(line) for line in open(cfg["eval"]["golden_set_path"])]

    finetuned_path = str(Path(cfg["data"]["processed_dir"]).parent / "models" / "bge-finetuned")
    base_model = SentenceTransformer(cfg["embeddings"]["text_model"])
    ft_model = SentenceTransformer(finetuned_path)

    base_recall = recall_at_k(base_model, products_df, golden_set, [10])["recall@10"]
    ft_recall = recall_at_k(ft_model, products_df, golden_set, [10])["recall@10"]
    print(f"base recall@10={base_recall:.4f}  finetuned recall@10={ft_recall:.4f}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    if ft_recall > base_recall:
        with mlflow.start_run(run_name="airflow_retrain_promoted"):
            mlflow.log_metric("recall_at_10", ft_recall)
            mlflow.log_metric("recall_at_10_uplift", ft_recall - base_recall)
            artifact_subdir = Path(finetuned_path).name  # e.g. "bge-finetuned"
            # log_artifacts (plural) uploads a whole directory's contents;
            # log_artifact (singular) is for a single file.
            mlflow.log_artifacts(finetuned_path, artifact_path=artifact_subdir)
            registered = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{artifact_subdir}",
                "shoptalk-text-embedding",
            )
            print(f"promoted new model version: {registered.version}")
        return "promoted"

    with mlflow.start_run(run_name="airflow_retrain_rejected"):
        mlflow.log_metric("recall_at_10", ft_recall)
        mlflow.log_metric("recall_at_10_uplift", ft_recall - base_recall)
    print("fine-tuned model did not beat the current model -- not promoted")
    return "rejected"


def task_trigger_deploy(**context):
    """Placeholder for triggering the CI/CD pipeline (design doc §7.3: "if
    better, register new model version in MLflow -> trigger CI/CD deploy").
    Wire this to a real trigger (e.g. `requests.post` to a GitHub Actions
    `repository_dispatch` webhook, using a repo PAT from Airflow
    Connections/Variables) once a deploy target is live; not called here
    since there's no running deployment to trigger against yet."""
    promote_result = context["ti"].xcom_pull(task_ids="evaluate_and_promote")
    if promote_result == "promoted":
        print("would trigger: gh api repos/Krishhs89/shoptalk-x/dispatches -f event_type=model_promoted")
    else:
        print("no promotion -- skipping deploy trigger")


with DAG(
    dag_id="shoptalk_x_retrain_embeddings",
    default_args=default_args,
    description="Data -> rebuild triplets -> fine-tune -> evaluate -> promote -> (trigger deploy)",
    schedule_interval="@weekly",  # or trigger manually on a drift alert
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["shoptalk-x", "retraining"],
) as dag:
    pull_data = PythonOperator(task_id="pull_data", python_callable=task_pull_data)
    rebuild_golden_set = PythonOperator(task_id="rebuild_golden_set", python_callable=task_rebuild_golden_set)
    finetune_embeddings = PythonOperator(task_id="finetune_embeddings", python_callable=task_finetune_embeddings)
    evaluate_and_promote = PythonOperator(task_id="evaluate_and_promote", python_callable=task_evaluate_and_promote)
    trigger_deploy = PythonOperator(task_id="trigger_deploy", python_callable=task_trigger_deploy)

    pull_data >> rebuild_golden_set >> finetune_embeddings >> evaluate_and_promote >> trigger_deploy
