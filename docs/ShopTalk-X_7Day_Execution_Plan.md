# ShopTalk-X: 7-Day Execution Plan (Half-time, ~4 hrs/day)

**How we work:** Claude generates code + configs for each module; you run them (local GPU / Kaggle), paste back errors/outputs; we debug and iterate. Each day ends with a working checkpoint — never a broken state overnight.

**Scope decisions (balanced priority):**
- KEEP: two-stage retrieval, CLIP image search, verification head, Docker+EC2 deployment, MLflow, Evidently drift check, EDA notebook, latency report, all docs
- SIMPLIFIED: Airflow → a single retraining DAG demoed locally; monitoring → FastAPI /metrics + Evidently report (skip full Prometheus/Grafana)
- STRETCH (Day 7 only if ahead): quantity validation, voice input, personalisation
- DROPPED: k3s/Kubernetes (document the steps, don't execute), LLM LoRA run (deliver code + docs only, per spec allowance)

---

## Day 1 — Data + EDA + Baseline retrieval (~4 hrs)
- Download ABO subset (~10k products: listings metadata + small images variant)
- Claude writes: preprocessing script (clean text, join fields), EDA notebook (distributions, missing values, category imbalance → modeling decisions)
- Embed descriptions with bge-base-en-v1.5 (local GPU, minutes) → Chroma index
- Baseline: text query → top-10 ANN results in a notebook
- **Checkpoint:** semantic text search working; EDA notebook 80% done

## Day 2 — Two-stage retrieval + golden set + MLflow (~4 hrs)
- Add cross-encoder reranker (ms-marco-MiniLM-L-6-v2), batched
- Claude writes: golden eval set generator (~100 queries via LLM + your manual spot-check), eval harness (Recall@K, NDCG@10 via ranx)
- MLflow up (docker-compose); log baseline vs reranked runs → uplift table
- **Checkpoint:** measured two-stage uplift; experiments tracked

## Day 3 — Multimodal: captions + CLIP image search (~4 hrs)
- BLIP captioning batch job (Kaggle or local GPU overnight if slow) → append captions, re-embed
- OpenCLIP embeddings for catalog images → second collection (or joint space)
- Image-as-query path: upload photo → CLIP embed → ANN → rerank
- **Checkpoint:** photo search working end-to-end in notebook

## Day 4 — RAG + LLM + FastAPI service (~4-5 hrs)
- Ollama with Llama-3.1-8B-Instruct 4-bit on local GPU
- Claude writes: LangChain RAG chain (retriever → rerank → prompt template → streaming LLM), conversation memory
- FastAPI service: /search/text, /search/image, /health, /metrics; models loaded at startup; latency middleware logging per-stage ms
- **Checkpoint:** curl the API, get grounded product recommendations with IDs

## Day 5 — Verification head + fine-tuning + UI (~4-5 hrs)
- Fine-tune bge embeddings with triplet loss + hard negatives mined from own index (Kaggle GPU, ~1-2 hrs) → log pre/post recall@K to MLflow
- Verification: Siamese pair dataset from multi-view ABO images → small MLP head → ROC-AUC, FAR/FRR
- Streamlit UI: text box, image upload, chat history, thumbs feedback, verification tab
- **Checkpoint:** full demo runs locally end-to-end

## Day 6 — Deployment + monitoring + latency report (~4-5 hrs)
- Dockerfile(s) + docker-compose (api + ollama + chroma); GitHub Actions workflow: test → build → push ECR
- Launch EC2 G4dn.xlarge, deploy, smoke test; **stop instance when done each session**
- Locust load test → P50/P95/P99 per stage table; quantization/rerank-depth tradeoff mini-study
- Evidently drift report (reference vs simulated shifted queries); simple retraining DAG (Airflow docker-compose, one DAG: data → finetune → eval → register)
- **Checkpoint:** live cloud demo + latency SLO table + drift report

## Day 7 — Docs, video, stretch (~4 hrs)
- Claude drafts: README, requirements.txt per component, architecture doc final pass, initial-results log from MLflow exports, model cards, presentation deck outline
- Record demo video (text, image, verification, deployment walkthrough)
- Stretch if ahead: quantity validation (YOLO count vs order), Whisper voice input (fast to add), personalisation rerank blend
- **Checkpoint:** submission checklist (§10b of design doc) fully ticked

---

## Daily rhythm
1. Start of session: tell Claude "Day N, ready" → Claude generates that day's code/files
2. You run, paste errors/outputs back → debug together
3. End of session: commit to GitHub, note carryover

## Risk valves
- Day 3 captioning slow → subsample to 5k products (fine per spec)
- Day 5 fine-tuning underwhelms → report the honest comparison (still full rubric credit for experimentation)
- Day 6 AWS friction → local Docker demo + documented AWS steps (spec allows demonstrating deployment via video)
- Behind by >1 day → drop verification head to stretch, keep two-stage multimodal RAG + deployment intact
