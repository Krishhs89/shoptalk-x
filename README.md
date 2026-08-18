# ShopTalk-X

A production-style multimodal shopping assistant with visual order
verification, built on the Amazon Berkeley Objects (ABO) dataset —
conversational text/photo search over a ~10k-product catalog, plus a
"did I get the right item?" photo-verification flow.

**Start here:**
- **Requirements, approach, and a start-to-finish build guide (read this to reproduce the whole project yourself)** → [docs/PROJECT_DEFINITION.md](docs/PROJECT_DEFINITION.md)
- **How do I use the running app?** → [docs/USAGE_WALKTHROUGH.md](docs/USAGE_WALKTHROUGH.md)
- **How does a request/build/deploy actually flow?** → [docs/WORKFLOW.md](docs/WORKFLOW.md)
- **What does it do?** → [docs/FUNCTIONAL_OVERVIEW.md](docs/FUNCTIONAL_OVERVIEW.md)
- **How is it built?** → [docs/TECHNICAL_ARCHITECTURE.md](docs/TECHNICAL_ARCHITECTURE.md)
- **Per-model details** → [docs/model_cards/](docs/model_cards/)
- **Original spec + design rationale** → [docs/ShopTalk-X_Production_Design_Document.md](docs/ShopTalk-X_Production_Design_Document.md)
- **Presentation outline** → [docs/PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md)
- **Real generative-answer examples, including a documented failure mode** → [docs/QUALITATIVE_EVALUATION.md](docs/QUALITATIVE_EVALUATION.md)
- **Demo video recording script** → [docs/VIDEO_RECORDING_SCRIPT.md](docs/VIDEO_RECORDING_SCRIPT.md)

Status: **all 7 build days complete, plus production hardening** (core
pipeline, RAG service, frontend, deployment/MLOps, docs, personalization
and voice-input stretch goals, and — as of the production-hardening pass —
a fine-tuned model wired into serving, an Airflow retraining DAG that has
actually run end-to-end and promoted a measurably better model, and
quantity/count validation via a pretrained YOLO detector, all verified
live in a real AWS EC2 deployment). **The AWS instance itself is currently
unreachable** as of 2026-08-13 — the training-lab AWS account lost API/SSH
access (see [docs/deployment/aws_ec2.md](docs/deployment/aws_ec2.md)'s
status note) — this is an external lab-account issue, not a defect in the
app; the local/offline Docker deployment
([docs/deployment/offline_production.md](docs/deployment/offline_production.md))
is unaffected and redeployment to AWS is a same-day task once fresh
credentials exist. See [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md)
for the full story — requirements, approach, tools, problems hit and how
they were solved, and what a real-world team would do next.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements/<component>.txt for just one piece
export PYTHONPATH=src

# Data -> embeddings -> index (see "Full pipeline" below for every step)
python -m shoptalk.data.download_abo
python -m shoptalk.data.preprocess
python -m shoptalk.data.caption_images
python -m shoptalk.embeddings.embed_text
python -m shoptalk.embeddings.embed_image

# LLM (separate terminal): install Ollama (https://ollama.com), then
ollama pull llama3.1:8b-instruct-q4_0

# Serve
uvicorn shoptalk.api.main:app --reload          # API on :8000, docs at :8000/docs
streamlit run src/shoptalk/ui/app.py            # UI on :8501

# ...or the whole stack in containers:
docker compose up -d --build
```

## What's built, by day

### Day 1 — data + EDA + baseline retrieval
- `shoptalk/data/download_abo.py`, `preprocess.py` — pulls an English-only
  ABO subset directly from its public S3 bucket, cleans/joins listing
  fields. **ABO has no price field** — `price_usd` is a synthetic,
  category-band-seeded value, clearly flagged `price_is_synthetic=True`
  everywhere, so "under $50"-style queries are demoable without pretending
  it's real Amazon pricing.
- `notebooks/01_eda.ipynb` — distributions, missing-value analysis,
  vocabulary analysis, NLP preprocessing rationale, insights tied to
  modeling decisions.
- `shoptalk/embeddings/embed_text.py`, `retrieval/search.py` — bi-encoder
  (`bge-base-en-v1.5`) embeddings + Chroma, baseline single-stage search.

### Day 2 — two-stage retrieval, golden eval set, MLflow
- `retrieval/rerank.py`, `two_stage.py` — cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) reranking on top of stage-1 ANN.
- `eval/generate_golden_set.py`, `evaluate_retrieval.py` — auto-generated,
  human-spot-checkable test set (`data/eval/golden_set.jsonl`); Recall/MRR/
  NDCG harness with MLflow logging.
- **Measured uplift** (`results/day2_retrieval_eval.md`): Recall@10 +10.5pp,
  MRR +11.3pp, NDCG@10 +14.6pp from reranking.

### Day 3 — multimodal: BLIP captions + CLIP image search
- `data/caption_images.py` — BLIP captions folded into the embedded text.
- `embeddings/embed_image.py`, `retrieval/image_search.py` — OpenCLIP image
  index + photo-as-query search, reranked via a BLIP-generated pseudo-text
  query (reuses the Day-2 reranker instead of a second model).
- `notebooks/02_image_search_demo.ipynb` — visual walkthrough.

### Day 4 — RAG + LLM + FastAPI backend
- `rag/prompts.py`, `chain.py` — LangChain retrieval → prompt → LLM chain
  (Ollama, `llama3.1:8b-instruct-q4_0`), with prompt-injection-resistant
  delimiting and explicit (non-LangChain-Memory) conversation history.
- `api/main.py` + `schemas.py`/`security.py`/`logging_store.py` — FastAPI
  service: `/search/text`, `/search/image`, `/verify`, `/verify/quantity`,
  `/feedback`, `/health`, `/metrics`; models loaded once at startup;
  API-key auth, rate limiting, per-request SQLite prediction logging.

### Day 5 — fine-tuning, verification head, frontend
- `embeddings/finetune_text.py`, `eval_finetune.py` — triplet-loss
  fine-tuning using same-category hard negatives mined from the golden
  set. **Measured uplift** (`results/day5_finetune_eval.md`): Recall@10
  0.897 → 0.959.
- `verification/*` — Siamese pairs from ABO multi-view images + hard
  negatives, small MLP head, match/mismatch/**suspect** (human-review band)
  verdicts. **Measured** (`results/day5_verification_eval.md`): ROC-AUC
  0.926, FAR 0.048, FRR 0.200 (smoke-test scale).
- `ui/app.py` — Streamlit frontend: chat search (text/photo/voice),
  conversation history, thumbs feedback, a separate order-verification tab.

### Day 6 — deployment + monitoring + MLOps
- `Dockerfile`, `docker/ui/`, `docker-compose.yml` — full local stack
  (Ollama + API + UI + MLflow).
- `.github/workflows/ci.yml` — lint (`ruff`) → test (`pytest`, 29 tests) →
  build/push to ECR → deploy to EC2, with cloud stages that skip cleanly
  (not fail) when credentials aren't configured.
- `docs/deployment/aws_ec2.md` — full EC2 deployment guide; **now live**
  (see "Production hardening" below) — originally documented-not-executed,
  actually run once real AWS access existed.
- `loadtest/locustfile.py`, `monitoring/drift_report.py` — Locust load
  testing; Evidently drift detection, validated to actually detect a
  simulated distribution shift (`results/day6_drift_report.md`).
- `airflow/dags/retrain_embeddings_dag.py` + `docker-compose.airflow.yml` —
  retraining DAG (pull data → rebuild triplets → fine-tune → evaluate →
  promote-if-better → trigger deploy); its MLflow promotion logic was
  validated directly against real trained models.

### Day 7 — docs, model cards, LoRA/QLoRA code
- `requirements/*.txt` — per-component dependency files (data, embeddings,
  eval, serving, ui, monitoring, voice, dev, llm_finetune), plus the
  aggregate root `requirements.txt`.
- `docs/model_cards/` — intended use, training data, eval results,
  limitations per model (embedding, reranker, CLIP/BLIP, LLM, verification).
- `llm_finetune/prepare_dataset.py`, `train_lora.py` — LoRA/QLoRA
  instruction fine-tuning; dataset generation **was run for real** against
  the live pipeline, the GPU training step is documented-not-executed per
  the problem statement's explicit allowance (see
  `docs/finetuning/llm_lora_qlora.md`).

### Stretch goals
- **Personalization** (`personalization/*`) — synthetic user interaction
  history, decayed profile vectors, rerank blend
  (`α·cross-encoder + β·profile-cosine`). Validated end-to-end with a
  measured hit-rate uplift (`results/stretch_personalization_eval.md`).
- **Voice input** (`voice/transcribe.py`) — faster-whisper STT wired into
  the UI's chat tab. Validated with a real generated speech clip
  transcribed correctly end-to-end.
- **Quantity/count validation** (`counting/*`) — `POST /verify/quantity` +
  a "Verify quantity" UI tab, checking a claimed item count against a
  photo via a **pretrained** YOLOv8n (COCO) detector — no custom-trained
  counting model, since there's no labeled counting dataset for this
  catalog. Honestly scoped: most catalog categories have no COCO
  equivalent and return `"unsupported"` rather than a guessed count (see
  `docs/model_cards/quantity_counting.md`).

### Production hardening (post-Day-7)
Closing the gaps between "demoed locally" and "actually running in
production":
- **Fine-tuned embedding model wired into serving** — `configs/config.yaml`
  now points live search at `data/models/bge-finetuned` (previously
  fine-tuning was validated in isolation but never actually served).
- **Live AWS deployment** — the full stack (API, UI, MLflow, Ollama) is
  running on a real EC2 `g4dn.xlarge` instance, not just documented (see
  `docs/deployment/aws_ec2.md`, now written from what was actually run,
  including the instance-level config — swap, file permissions — that
  isn't in git).
- **Retraining DAG run end-to-end for real** on that instance, past a
  genuine multi-cause debugging saga (memory exhaustion → in-process
  execution memory leak → captured subprocess output → file permissions);
  see the "Operational lessons" section of `docs/deployment/aws_ec2.md`.
- **Quantity/count validation** added (see above) — the last of the
  originally-scoped-out stretch items.
- **Offline vs. online production docs split**:
  `docs/deployment/offline_production.md` (single-machine Docker) and
  `docs/deployment/aws_ec2.md` (AWS EC2), so either deployment target has
  an accurate, dedicated runbook.

## Full pipeline (every command, in order)

```bash
export PYTHONPATH=src

# --- Day 1: data ---
python -m shoptalk.data.download_abo              # ~10-20 min
python -m shoptalk.data.preprocess
# open notebooks/01_eda.ipynb and run all cells

# --- Day 3: captions (before first embed, so captions get indexed) ---
python -m shoptalk.data.caption_images

# --- Day 1/3: embeddings + indexes ---
python -m shoptalk.embeddings.embed_text
python -m shoptalk.embeddings.embed_image

# --- Day 2: two-stage retrieval + eval ---
python -m shoptalk.retrieval.two_stage --query "red shirt for men under 50 dollars"
python -m shoptalk.eval.generate_golden_set
#   -> spot-check data/eval/golden_set_review.csv, hand-edit the .jsonl if needed
docker compose up -d mlflow                        # optional, UI at :5001 (5000 is taken by macOS AirPlay)
python -m shoptalk.eval.evaluate_retrieval

# --- Day 3: photo search ---
python -m shoptalk.retrieval.image_search --image path/to/photo.jpg

# --- Day 5: fine-tuning + verification ---
python -m shoptalk.embeddings.finetune_text
python -m shoptalk.embeddings.eval_finetune --finetuned-path data/models/bge-finetuned
python -m shoptalk.verification.build_pairs
python -m shoptalk.verification.train_verification

# --- Day 4: LLM + API (separate terminal for Ollama) ---
ollama serve &
ollama pull llama3.1:8b-instruct-q4_0
uvicorn shoptalk.api.main:app --reload              # :8000, Swagger at /docs
python -m shoptalk.rag.chain --query "..." --stream

# --- Day 5: UI ---
streamlit run src/shoptalk/ui/app.py                # :8501

# --- Day 6: load test + drift + retraining DAG ---
locust -f src/shoptalk/loadtest/locustfile.py --host http://localhost:8000 \
  --users 10 --spawn-rate 2 --run-time 5m --headless --csv results/locust
python -m shoptalk.monitoring.drift_report
docker compose -f docker-compose.airflow.yml up --build   # :8080

# --- Stretch: personalization + voice ---
python -m shoptalk.personalization.simulate_interactions
python -m shoptalk.personalization.evaluate_personalization
python -m shoptalk.voice.transcribe --audio path/to/clip.wav

# --- Day 7: LLM instruction dataset (LoRA training itself needs a GPU) ---
python -m shoptalk.llm_finetune.prepare_dataset --n 20
```

Config (dataset size, all model names/params, paths) lives entirely in
`configs/config.yaml`.

## Project layout

```
configs/                pipeline configuration (single source of truth)
data/raw/, processed/     downloaded + cleaned catalog (gitignored)
data/chroma/               vector indexes (gitignored)
data/eval/                golden eval set (committed -- required deliverable)
data/verification/         trained MLP head (committed) + pairs (gitignored)
results/                  evaluation output tables (committed)
requirements/             per-component dependency files
src/shoptalk/
  data/                    download + preprocessing + captioning
  embeddings/              text/image embedding, fine-tuning, eval
  retrieval/               stage-1 search, rerank, two-stage, image search
  eval/                    golden set generation, retrieval evaluation
  rag/                     LangChain RAG chain + prompts
  api/                     FastAPI service
  ui/                      Streamlit frontend
  verification/            visual order verification
  personalization/         stretch: rerank personalization
  voice/                   stretch: Whisper STT
  llm_finetune/             LoRA/QLoRA dataset + training code
  monitoring/               Evidently drift report
  loadtest/                 Locust load test
notebooks/               EDA + image search demo + Colab/Kaggle GPU pipeline
docs/                    all documentation (see top of this file)
docs/model_cards/         per-model cards
docs/deployment/           AWS EC2 guide
docs/finetuning/            LoRA/QLoRA guide
airflow/dags/             retraining DAG
tests/                   pytest unit tests (29, run in CI)
.github/workflows/         CI/CD
```

## Testing

```bash
pip install -r requirements/dev.txt
ruff check src/ tests/
PYTHONPATH=src pytest tests/ -v
```
