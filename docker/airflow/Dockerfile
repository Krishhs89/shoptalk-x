# Minimal local-demo Airflow image: base Airflow + this project's package so
# DAG tasks can `from shoptalk... import ...` directly (see retrain_embeddings_dag.py).
FROM apache/airflow:2.10.3-python3.11

USER root
RUN mkdir -p /opt/shoptalk && chown airflow /opt/shoptalk
USER airflow

COPY --chown=airflow requirements.txt /opt/shoptalk/requirements.txt
# Airflow pins its own dependency versions; installing the full ML stack
# alongside it can conflict, so this demo installs only what the DAG's task
# functions actually import (pandas/sentence-transformers/mlflow/etc.),
# skipping heavier serving-only deps (fastapi, streamlit, locust, evidently).
RUN pip install --no-cache-dir --user \
    pandas==2.2.3 pyarrow==17.0.0 PyYAML==6.0.2 \
    sentence-transformers==3.3.1 chromadb==0.5.23 \
    mlflow==2.18.0 datasets==3.1.0 accelerate==1.1.1

COPY --chown=airflow configs/ /opt/shoptalk/configs/
COPY --chown=airflow src/ /opt/shoptalk/src/
ENV PYTHONPATH=/opt/shoptalk/src
