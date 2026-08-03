# ShopTalk-X

Production multimodal shopping assistant with visual verification, built on the Amazon Berkeley Objects (ABO) dataset. Full design rationale: [docs/ShopTalk-X_Production_Design_Document.md](docs/ShopTalk-X_Production_Design_Document.md). Build schedule: [docs/ShopTalk-X_7Day_Execution_Plan.md](docs/ShopTalk-X_7Day_Execution_Plan.md).

Status: **Day 3 — multimodal: BLIP captions + CLIP image search.**

## What's here

### Day 1 — data + EDA + baseline retrieval
- `src/shoptalk/data/download_abo.py` — pulls an English-language ~10k-product subset of ABO directly from its public S3 bucket (listings metadata + main product images), no AWS credentials required.
- `src/shoptalk/data/preprocess.py` — cleans/joins listing fields into a flat product table (`data/processed/products.parquet`). Note: **ABO has no price field** — `price_usd` is a synthetic, category-band-seeded value clearly flagged with `price_is_synthetic=True`, added only so price-filter queries ("under $50") are demoable. It is never real Amazon pricing.
- `notebooks/01_eda.ipynb` — category/brand/price distributions, missing-value analysis, description-length & vocabulary analysis, NLP preprocessing rationale, image availability, and modeling decisions the EDA drives (e.g. category imbalance → within-category hard-negative mining).
- `src/shoptalk/embeddings/embed_text.py` — embeds each product's joined text with `BAAI/bge-base-en-v1.5`, indexes into a persistent Chroma collection.
- `src/shoptalk/retrieval/search.py` — baseline single-stage text query → top-K ANN search over the Chroma index.

### Day 2 — two-stage retrieval, golden eval set, MLflow
- `src/shoptalk/retrieval/rerank.py` — cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranker, batched.
- `src/shoptalk/retrieval/two_stage.py` — stage-1 ANN top-100 → stage-2 cross-encoder rerank → top-K.
- `src/shoptalk/eval/generate_golden_set.py` — generates ~100 (query → relevant products) pairs, template-based and grounded in real sampled products, stratified across categories. Filters out unclean brand/color values (leaked internal codes, mixed-script text) before using them in a query. Writes `data/eval/golden_set.jsonl` (source of truth) + `golden_set_review.csv` (for the human spot-check the spec calls for).
- `src/shoptalk/eval/evaluate_retrieval.py` — runs the golden set through both stage-1-only and full two-stage retrieval, computes Recall@10/50/100, MRR, NDCG@10 via `ranx`, writes an uplift table to `results/day2_retrieval_eval.md`, and logs both runs to MLflow.
- `docker-compose.yml` + `docker/mlflow/` — MLflow tracking server (`docker compose up mlflow`, UI at `localhost:5000`). Works without it too — defaults to a local `file:./mlruns` store.

### Day 3 — multimodal: captions + CLIP image search
- `src/shoptalk/data/caption_images.py` — BLIP (`blip-image-captioning-base`) batch-captions every product's main image, appends a `caption` column and folds it into the `document` field so future re-embeddings pick up visual attributes (pattern, shape, material) sellers often omit from bullet points. Uses `repetition_penalty`/`no_repeat_ngram_size` — without them BLIP occasionally degenerates into a repeated-token loop (caught this on a real image during validation).
- `src/shoptalk/embeddings/embed_image.py` — OpenCLIP (`ViT-B-32`, LAION `laion2b_s34b_b79k`) embeds every catalog image into a **second** Chroma collection (`shoptalk_images`) — a separate embedding space from the text collection, not directly comparable.
- `src/shoptalk/retrieval/image_search.py` — photo-as-query: CLIP-embed the uploaded photo → ANN over the image collection → **BLIP captions the query photo itself** to manufacture a pseudo-text query → reruns the exact same Day-2 cross-encoder reranker against the candidates. (The cross-encoder is text-only and a photo has no natural query to pair it with, so this reuses Day 2's reranker instead of needing a separate image-reranking model.)
- `notebooks/02_image_search_demo.ipynb` — the Day-3 checkpoint: runs a query photo through the full pipeline and displays the query image alongside its top-K visual matches.

All scripts validated end-to-end against a live sample pulled from the real ABO bucket (see commit history) — they aren't just sketches. The Day-2 validation run (300 products, 30 golden queries) showed a genuine two-stage uplift: Recall@10 +10.5%, MRR +11.3%, NDCG@10 +14.6%, with Recall@100 unchanged (~0%) as expected, since reranking only reorders the stage-1 candidate pool rather than expanding it. Day 3's image search was validated with a real product photo as the query: CLIP correctly recovered the exact source product at similarity 1.000, BLIP produced a specific, accurate pseudo-query ("a pair of brown sued shoes"), and reranking kept the true match at rank 1 with a wide score margin over visually-similar alternatives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

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

# 5. Baseline (stage-1-only) query
python -m shoptalk.retrieval.search --query "red shirt for men under 50 dollars" --top-k 10

# 6. Two-stage query (stage-1 ANN -> cross-encoder rerank)
python -m shoptalk.retrieval.two_stage --query "red shirt for men under 50 dollars" --top-k 10

# 7. Generate + spot-check the golden eval set
python -m shoptalk.eval.generate_golden_set
#   -> open data/eval/golden_set_review.csv, sanity-check a sample of
#      queries/positives, hand-edit data/eval/golden_set.jsonl if anything
#      looks wrong, then treat it as frozen ground truth

# 8. (optional) start MLflow's tracking server + UI
docker compose up -d mlflow   # UI at http://localhost:5000

# 9. Evaluate stage-1 vs two-stage on the golden set, log both runs to MLflow
python -m shoptalk.eval.evaluate_retrieval

# 10. BLIP-caption every product image, fold captions into the document field
python -m shoptalk.data.caption_images

# 11. Re-embed text (now caption-augmented) and CLIP-embed catalog images
python -m shoptalk.embeddings.embed_text     # overwrites the text collection with captioned documents
python -m shoptalk.embeddings.embed_image    # builds the separate image collection

# 12. Photo-as-query search
python -m shoptalk.retrieval.image_search --image path/to/photo.jpg --top-k 10
#   or open notebooks/02_image_search_demo.ipynb for the visual walkthrough
```

Config (dataset size, model names, rerank/eval/mlflow settings, paths) lives in `configs/config.yaml`.

## Project layout

```
configs/            pipeline configuration
data/raw/            downloaded ABO listings + images (gitignored)
data/processed/       cleaned product table (gitignored)
data/chroma/          vector index (gitignored)
data/eval/           golden eval set (committed -- required submission artifact)
results/             evaluation output tables (committed)
docker/mlflow/        MLflow tracking server image
src/shoptalk/data/     download + preprocessing
src/shoptalk/embeddings/  embedding generation
src/shoptalk/retrieval/   stage-1 search, cross-encoder rerank, two-stage pipeline
src/shoptalk/eval/       golden set generation, retrieval evaluation
notebooks/           EDA and experimentation notebooks
docs/                problem statement, design doc, execution plan, session log
```

## Roadmap

See [docs/ShopTalk-X_7Day_Execution_Plan.md](docs/ShopTalk-X_7Day_Execution_Plan.md) for Day 4 (RAG + LLM + FastAPI service) through Day 7 (docs, video, stretch goals).
