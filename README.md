# ShopTalk-X

Production multimodal shopping assistant with visual verification, built on the Amazon Berkeley Objects (ABO) dataset. Full design rationale: [docs/ShopTalk-X_Production_Design_Document.md](docs/ShopTalk-X_Production_Design_Document.md). Build schedule: [docs/ShopTalk-X_7Day_Execution_Plan.md](docs/ShopTalk-X_7Day_Execution_Plan.md).

Status: **Day 1 — data + EDA + baseline retrieval.**

## What's here (Day 1)

- `src/shoptalk/data/download_abo.py` — pulls an English-language ~10k-product subset of ABO directly from its public S3 bucket (listings metadata + main product images), no AWS credentials required.
- `src/shoptalk/data/preprocess.py` — cleans/joins listing fields into a flat product table (`data/processed/products.parquet`). Note: **ABO has no price field** — `price_usd` is a synthetic, category-band-seeded value clearly flagged with `price_is_synthetic=True`, added only so price-filter queries ("under $50") are demoable. It is never real Amazon pricing.
- `notebooks/01_eda.ipynb` — category/brand/price distributions, missing-value analysis, description-length & vocabulary analysis, NLP preprocessing rationale, image availability, and modeling decisions the EDA drives (e.g. category imbalance → within-category hard-negative mining).
- `src/shoptalk/embeddings/embed_text.py` — embeds each product's joined text with `BAAI/bge-base-en-v1.5`, indexes into a persistent Chroma collection.
- `src/shoptalk/retrieval/search.py` — baseline single-stage text query → top-K ANN search over the Chroma index.

All five scripts have been validated end-to-end against a live 150-product sample pulled from the real ABO bucket (see commit history) — they aren't just sketches.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Day-1 pipeline

```bash
export PYTHONPATH=src

# 1. Download ~10k English-language products + main images (~10-20 min depending on network)
python -m shoptalk.data.download_abo
#   smoke test first if you want: python -m shoptalk.data.download_abo --limit 200

# 2. Clean/join fields into data/processed/products.parquet + .csv
python -m shoptalk.data.preprocess

# 3. Open notebooks/01_eda.ipynb and run all cells (jupyter notebook / jupyter lab)

# 4. Embed + index (GPU recommended but CPU/MPS works for this subset size)
python -m shoptalk.embeddings.embed_text

# 5. Query it
python -m shoptalk.retrieval.search --query "red shirt for men under 50 dollars" --top-k 10
```

Config (dataset size, model names, paths) lives in `configs/config.yaml`.

## Project layout

```
configs/            pipeline configuration
data/raw/            downloaded ABO listings + images (gitignored)
data/processed/       cleaned product table (gitignored)
data/chroma/          vector index (gitignored)
src/shoptalk/data/     download + preprocessing
src/shoptalk/embeddings/  embedding generation
src/shoptalk/retrieval/   search / retrieval
notebooks/           EDA and experimentation notebooks
docs/                problem statement, design doc, execution plan, session log
```

## Roadmap

See [docs/ShopTalk-X_7Day_Execution_Plan.md](docs/ShopTalk-X_7Day_Execution_Plan.md) for Day 2 (cross-encoder reranking, golden eval set, MLflow) through Day 7 (docs, video, stretch goals).
