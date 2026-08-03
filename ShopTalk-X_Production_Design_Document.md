# ShopTalk-X: Production Multimodal Shopping Assistant with Visual Verification

**Design Document v1.0** | Krishna | August 2026

---

## 1. Problem Statement

Build a production-grade e-commerce assistant that:

1. **Discovers** — accepts natural-language text ("show me red shirts under $50, but not striped") or a real-world photo, and returns the most relevant products with an LLM-generated natural-language response.
2. **Verifies** — given a photo of a received item and an order ID, visually verifies the item matches the ordered product (wrong-item / counterfeit detection).

The system must be production-ready: two-stage low-latency retrieval, containerized deployment, monitoring, drift-triggered retraining, and security/governance controls.

**Business impact:** conversational search improves product discoverability and NPS; visual verification attacks returns fraud and wrong-item shipments — a multi-billion-dollar cost center in e-commerce.

---

## 2. Data

| Item | Detail |
|---|---|
| Dataset | Amazon Berkeley Objects (ABO) — subset of ~10k products |
| Fields used | Product images, text description, keywords, category, brand, price |
| Augmented text | BLIP image captions appended to descriptions (captures visual attributes not in text) |
| Derived data | Verification pairs: positives = multi-view images of same product; hard negatives = nearest neighbors from vector DB (same category, different product) |
| Robustness set | Augmented query images (blur, occlusion, glare via albumentations) for stress-testing retrieval |

**Preprocessing pipeline:** clean/normalize text → batch BLIP captioning (one-time Kaggle GPU job) → build joint documents (description + keywords + caption) → embed → index.

**EDA (graded deliverable, 15%):** dedicated notebook covering category/brand/price distributions, description length & vocabulary analysis, image count per product, missing-value analysis, caption-vs-description overlap, and NLP preprocessing rationale (tokenization, normalization choices). Visualizations for each; insights explicitly tied to modeling decisions (e.g., category imbalance → hard-negative sampling strategy).

---

## 3. Model Architecture

### 3.1 Embedding models (offline, precomputed)

- **Text bi-encoder:** open-source sentence embedding model (e.g., bge/e5 family), fine-tuned on ABO with **triplet loss** and hard-negative mining (same category, different product). Compare pre-trained vs fine-tuned via inter/intra-category cosine separation + recall@K. Track in MLflow.
- **Joint image-text space:** **CLIP** embeddings for all catalog images, enabling photo-as-query. Optionally fine-tune CLIP head with LoRA on ABO category pairs.
- All 10k product vectors **precomputed** and stored in the vector DB (Chroma/Milvus). Nothing is embedded at index time during a request except the user's query.

### 3.2 Two-stage retrieval (per instructor notes)

**Stage 1 — Fast retrieval (low latency):** reduce search space 10k → top-100.
- Compute embedding for user input (text or cropped image).
- ANN search (HNSW) in vector DB → top-100 candidates.
- Target latency: **< 50 ms**.

**Stage 2 — Precise reranking (higher latency, higher precision):** top-100 → top-K (5–10).
- **Cross-encoder** scores each (user query, item description) pair jointly: (query, item_1) → score_1 … (query, item_100) → score_100.
- Cross-encoders are far more accurate than bi-encoders because query and document attend to each other, but too slow to run over 10k items — hence only on the top-100.
- Model: ms-marco MiniLM cross-encoder or similar; batch the 100 pairs on GPU.
- Target latency: **< 300 ms**.

**Stage 3 — Generation:**
- Top-K reranked products + their metadata → RAG prompt → self-hosted quantized LLM (7–8B, e.g., Llama-3.1-8B-Instruct, 4-bit) → natural-language recommendation with product IDs/links.
- RAG pipeline orchestrated with **LangChain** (per spec): retriever wrapper over the vector DB + reranker, prompt templates, conversation memory for follow-up queries.
- Target latency: **< 2.5 s** (streaming tokens so perceived latency is lower).

### 3.2b LLM fine-tuning (deliverable: code + documented steps)

- Provide **LoRA/QLoRA instruction fine-tuning code** for the generation LLM (PEFT + bitsandbytes), with dataset format spec (instruction/query/retrieved-context/response), training config, and merge/serve steps — runnable on Kaggle GPU.
- Execute a small demonstration run on a synthetic e-commerce instruction set (LLM-generated Q&A pairs over ABO products) and compare against the base model on the golden set; document LoRA-vs-QLoRA memory/quality tradeoff.

### 3.2c Personalisation (optional deliverable — implemented as reranking)

- Simulate user–item interaction history (clicks/purchases per synthetic user).
- Build a user profile vector = decayed average of embeddings of interacted products.
- **Post-retrieval rerank blend:** final score = α·cross-encoder score + β·cosine(user profile, item embedding); α/β tuned on held-out interactions. Evaluate uplift in Precision@K for repeat users vs non-personalized baseline.

### 3.3 Computer-vision subsystems (the CV depth)

**a) In-the-wild image ingestion:**
- Fine-tuned **YOLOv8** detector localizes the product in cluttered user photos; crop before CLIP embedding.
- Annotation: few hundred images, SAM-assisted labeling in LabelStudio.
- Eval: mAP@50; ablation of retrieval recall@K with vs without detection-cropping on degraded query images.

**b) Visual verification head:**
- Siamese comparison on fine-tuned embeddings: embed(received-item photo) vs embed(catalog images for ordered ASIN).
- Small MLP classifier on embedding pairs → match / mismatch / suspect, with calibrated confidence.
- Eval: ROC-AUC, FAR/FRR at chosen operating threshold; threshold selection documented.

**c) Quantity validation (BinSense essence — order fulfillment check):**
- Extend verification to full order validation: given an order (items + quantities) and a photo, confirm **each item AND its quantity** is present — e.g., "2× Nerf blaster, 1× projector: all present?"
- Pipeline: YOLO detects all product instances in the photo → each crop embedded and matched to ordered ASINs via the verification head → per-ASIN instance counting → compare against ordered quantities → per-line-item verdict (✓ / ✗ / uncertain).
- Handles BinSense's hard cases explicitly: occluded instances (flag "uncertain" below detection-confidence threshold rather than guessing), duplicate-looking products (hard-negative-trained embeddings disambiguate), partial visibility (test-time augmentation — multi-crop inference).
- Evaluation: per-item quantity accuracy, order-level exact-match rate, and confusion analysis by occlusion level; report calibration of the "uncertain" band (fraction routed to human review vs error rate caught).
- UI: order-builder screen (select items + quantities, BinSense-style) → upload/select photo → per-line-item validation results with bounding-box overlays.

**d) Robustness:**
- Albumentations-based degradation suite (blur, occlusion, glare) in training and as a standing evaluation set.

---

## 4. Inference Service & Latency Engineering

### 4.1 Serving design

- **FastAPI** service, Dockerized; models loaded **once at service startup**, never per-request.
- Same preprocessing/transformers used in training are serialized and reused at inference (no refitting — prevents train/serve skew).
- Endpoints: `POST /search/text`, `POST /search/image`, `POST /verify`, `GET /health`, `GET /metrics`.
- LLM served via vLLM or Ollama (quantized) on the same G4dn.xlarge; embedding + cross-encoder models on GPU with dynamic batching.
- Streaming responses (SSE) for the LLM stage.

### 4.2 Latency budget (measure and report P50/P95/P99 per stage)

| Stage | Target P95 |
|---|---|
| Query embedding | 30 ms |
| ANN retrieval (top-100) | 50 ms |
| Cross-encoder rerank (100 pairs, batched) | 300 ms |
| LLM generation (first token) | 800 ms |
| LLM generation (complete) | 2.5 s |
| End-to-end (search) | **< 3 s** |
| Verification endpoint | < 500 ms |

**Optimization levers (document experiments):** LLM quantization level (4/8-bit), rerank depth (100 vs 50), embedding model size, HNSW ef_search parameter, caption cache, KV-cache reuse, batch sizes. Report a latency-vs-quality tradeoff curve — this is prime interview material.

---

## 5. Deployment (Real-World)

| Layer | Choice |
|---|---|
| Experimentation | Kaggle (35 GPU-hr/week) / Colab; all runs logged to MLflow |
| Packaging | Docker image per service (API, LLM server); docker-compose for local, single Dockerfile deliverable |
| Hosting | AWS EC2 **G4dn.xlarge** (single T4) for inference; stop when idle |
| Orchestration (stretch) | Kubernetes (EKS or k3s on EC2) with liveness/readiness probes, rolling updates |
| CI/CD | **GitHub Actions** (2000 free min): on push → lint/test → build Docker image → push to ECR → deploy to EC2/K8s. Remote-triggerable jobs |
| API docs | Swagger/OpenAPI auto-generated by FastAPI |
| UI | Streamlit/Gradio frontend hitting the REST API; supports text query, image upload, **voice input (Whisper STT → text query; optional gTTS spoken response)**, conversation history with follow-up support, thumbs up/down feedback buttons, product IDs/links rendered with results, verification flow |

**Deployment rules:** model artifacts pulled from MLflow model registry by version tag (staging → production promotion); zero-downtime rolling restart; rollback = redeploy previous image tag.

---

## 6. Model Monitoring

| What | How |
|---|---|
| System metrics | Prometheus + Grafana (or CloudWatch): request rate, error rate, per-stage latency P50/P95/P99, GPU utilization |
| Prediction logging | Every request: query, retrieved IDs, rerank scores, LLM output hash, latency breakdown, user feedback — logged to a store (SQLite/Postgres/S3 parquet) |
| Data drift | **Evidently**: distribution of query embeddings (PSI/KS on projected dims), query length, image-quality stats, category distribution of retrieved items vs reference window |
| Model drift | Rolling retrieval precision on a labeled golden set replayed weekly; rerank score distribution shift; verification FAR/FRR on audit samples |
| LLM quality | Ragas-style LLM-as-judge on sampled responses: faithfulness (grounded in retrieved products), relevance, helpfulness; thumbs-up/down feedback rate |
| Alerting | Threshold alerts on drift metrics, P95 latency, error rate → notification + retraining-pipeline trigger |

---

## 7. Continuous Improvement (Retraining Loop)

1. **Feedback capture:** explicit (thumbs up/down on responses, "wrong item" flags on verification) + implicit (click-through on recommended products).
2. **Golden evaluation set:** curated query→relevant-products test set; every model/prompt change is evaluated against it before promotion (regression gate in CI).
3. **Retraining triggers:** scheduled (monthly) OR drift-alert-driven. Orchestrated with **Airflow**: DAG = pull new data → rebuild pairs → fine-tune embeddings → evaluate vs golden set → if better, register new model version in MLflow → trigger CI/CD deploy.
4. **New products:** incremental embedding + index upsert pipeline (no full reindex); drift monitor flags novel-category queries.
5. **Prompt iteration:** prompts version-controlled; A/B compare via LLM-judge scores on the golden set.

---

## 8. Security & Governance

**Security**
- API authentication (API keys/JWT), rate limiting, request size caps (image uploads), HTTPS only.
- **Prompt-injection defenses:** retrieved product text is data, not instructions — structured prompt template with delimiters; output constrained to catalog products (no free-web claims); deny-list on system-prompt exfiltration attempts.
- Input validation/sanitization on all endpoints; image type/size checks; no execution of user content.
- Secrets in environment/SSM — never in code or images; least-privilege IAM roles; security groups restrict ports.
- Dependency scanning (pip-audit / GitHub Dependabot) in CI.

**Governance**
- **Privacy:** user photos processed transiently, not retained beyond logging window without consent; PII scrubbing in logs; deletion policy documented.
- **Model cards** for each model (embedding, reranker, detector, verifier, LLM): intended use, training data, eval results, limitations, failure modes.
- **Data licensing:** ABO is public/research-friendly; document license compliance.
- **Auditability:** every prediction traceable (request ID → model versions → retrieved docs → output); model registry preserves lineage.
- **Bias/fairness check:** verify retrieval quality is consistent across product categories and price bands; report per-slice recall.
- **Responsible LLM use:** grounding requirement (answers cite retrieved products), refusal behavior for out-of-domain queries, human-review path for verification "suspect" verdicts (a human confirms before any counterfeit accusation).

---

## 9. Evaluation Plan (Summary)

| Component | Metrics |
|---|---|
| Stage-1 retrieval | Recall@10/50/100, MRR on golden set |
| Stage-2 rerank | Precision@5/@10, NDCG@10; uplift vs no-rerank |
| Embedding fine-tune | Inter/intra-category cosine separation; recall@K pre vs post |
| Detector | mAP@50; retrieval recall uplift on cluttered queries |
| Verification | ROC-AUC, FAR/FRR at threshold, calibration curve |
| Quantity validation | Per-item count accuracy, order-level exact-match rate, occlusion-stratified confusion analysis, human-review routing rate |
| LLM/RAG | Faithfulness, answer relevance (Ragas/LLM-judge); qualitative review |
| System | Per-stage and E2E latency P50/P95/P99; throughput; error rate |
| Robustness | All retrieval metrics on degraded-image query set |

---

## 10. Milestones (build order = always-working demo)

1. **Week 1:** Data prep + EDA; BLIP captioning batch job; baseline text embeddings + vector DB; simple RAG with hosted LLM; Streamlit skeleton. *(Working text-search demo.)*
2. **Week 2:** Two-stage retrieval — add cross-encoder reranker; golden eval set; MLflow tracking; latency instrumentation. *(Instructor-notes architecture complete.)*
3. **Week 3:** CV — CLIP image search; YOLO detect/crop fine-tune; robustness augmentations + ablations. *(Multimodal demo.)*
4. **Week 4:** Verification head + calibration; self-host quantized LLM on EC2; Docker + FastAPI service; latency optimization to budget. *(Production service.)*
5. **Week 5:** CI/CD (GitHub Actions), monitoring (Prometheus/Grafana + Evidently), Airflow retraining DAG, security hardening. *(MLOps spine.)*
6. **Week 6:** Optional deliverables — voice input (Whisper), personalisation reranking, LLM LoRA/QLoRA fine-tuning run; documentation, model cards, demo video, presentation; stretch: K8s deploy, A/B prompt eval.

---

## 10b. Submission Checklist (maps to ShopTalk submission guidelines)

- [ ] All Colab/Kaggle notebooks (EDA, captioning, embedding fine-tuning, cross-encoder experiments, LLM LoRA/QLoRA fine-tuning)
- [ ] Fine-tuned model artifacts (embedding model, YOLO weights, verification head, LoRA adapters) + MLflow registry export
- [ ] Preprocessed dataset + documentation of the preprocessing process
- [ ] AWS inference code (FastAPI service, LangChain RAG pipeline) with documentation
- [ ] `requirements.txt` per component + top-level README with setup/run instructions for non-experts
- [ ] Dockerfile(s) + step-by-step deployment guide (ECR push, EC2 run, CI/CD trigger)
- [ ] System architecture & implementation document (this doc, updated with final decisions)
- [ ] Initial results & experiments log (metrics tables from MLflow, latency reports)
- [ ] Clear video recording of the working bot (text, image, voice, verification flows) + deployment/monitoring walkthrough
- [ ] Test dataset (query → relevant products) used for precision testing, with results
- [ ] Presentation deck for the expert-panel demo

---

## 11. Resume Bullets (draft)

- Built a production multimodal e-commerce assistant (text + image search over 10k products) using a two-stage retrieval architecture — ANN vector search (top-100, <50 ms) followed by cross-encoder reranking — improving Precision@10 by X% over single-stage retrieval.
- Fine-tuned sentence-embedding and CLIP models with triplet loss and hard-negative mining; fine-tuned YOLOv8 for product localization in cluttered user photos (mAP@50 = X), lifting retrieval recall on degraded images by X%.
- Designed a visual order-verification system (Siamese embedding comparison, ROC-AUC = X) to detect wrong-item/counterfeit deliveries, with calibrated thresholds and human-in-the-loop review.
- Deployed a self-hosted quantized LLM RAG service (FastAPI, Docker, AWS EC2) meeting a <3 s P95 end-to-end latency budget; instrumented per-stage P95/P99 dashboards.
- Implemented full MLOps loop: MLflow registry, GitHub Actions CI/CD, Evidently drift detection on query-embedding distributions, and Airflow-orchestrated drift-triggered retraining.
- Applied LLM security & governance controls: prompt-injection defenses, grounded generation, model cards, per-slice fairness evaluation, and auditable prediction lineage.

*(Replace X with your measured numbers.)*

---

## 12. Industry-Standard Stack (all free / open-source / free-tier)

The exact tools, models, methods, and evaluation approaches used in production ML teams — chosen so the entire project runs at near-zero cost.

### 12.1 Models (open-source, free)

| Purpose | Industry choice (free) | Notes |
|---|---|---|
| Text embeddings | **BAAI/bge-base-en-v1.5** or **intfloat/e5-base-v2** (HuggingFace) | Top of MTEB leaderboard among small open models; industry default for retrieval |
| Cross-encoder reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2**; upgrade: **BAAI/bge-reranker-base** | Standard rerankers used behind many production search stacks |
| Image-text joint space | **OpenCLIP ViT-B/32** (LAION) | Open reimplementation of CLIP; industry norm for multimodal retrieval |
| Image captioning | **Salesforce/BLIP** (or BLIP-2 if GPU allows) | De-facto open captioner |
| Object detection | **YOLOv8n/s (Ultralytics)** | Ubiquitous in industry CV; free for this use |
| Annotation assist | **SAM (Meta)** + **Label Studio** | Standard semi-automated labeling combo |
| LLM (generation) | **Llama-3.1-8B-Instruct** or **Mistral-7B-Instruct**, 4-bit GGUF/AWQ | The open-weights production workhorses |
| Speech-to-text (voice input) | **Whisper (openai/whisper-base)** or faster-whisper | Industry default STT, runs on CPU |
| Fine-tuning | **HuggingFace PEFT (LoRA/QLoRA) + bitsandbytes + TRL** | The standard open fine-tuning stack |

### 12.2 Infrastructure & serving tools (free tiers)

| Purpose | Industry choice (free) | Notes |
|---|---|---|
| Vector DB | **Milvus** (Docker, self-hosted) or **Qdrant**; Chroma for prototyping | All open-source; Qdrant/Milvus are what companies actually deploy |
| ANN algorithm | **HNSW** (built into all three) + **FAISS** for offline experiments | FAISS = Meta's library, industry benchmark standard |
| LLM serving | **vLLM** (GPU) or **llama.cpp/Ollama** (CPU/small GPU) | vLLM's continuous batching is the production norm |
| API framework | **FastAPI** + Pydantic validation + Uvicorn | Dominant Python ML-serving framework |
| Orchestration of RAG | **LangChain** (spec requirement); know that many teams also hand-roll | |
| Containers | **Docker + docker-compose**; **k3s** for a free single-node Kubernetes | k3s gives you a real K8s story at $0 |
| Experiment tracking | **MLflow** (self-hosted, free) — tracking + model registry | Industry standard alongside W&B (W&B free tier also fine) |
| Pipeline orchestration | **Apache Airflow** (Docker Compose) | Spec-aligned; industry standard |
| CI/CD | **GitHub Actions** (2000 free min/month) | Replaces Jenkins at most startups |
| Monitoring — system | **Prometheus + Grafana** (OSS, self-hosted) | The default metrics/dashboards pair |
| Monitoring — ML drift | **Evidently AI** (OSS) | Free, industry-recognized drift detection (PSI, KS tests) |
| Logging | **Python structlog → SQLite/Postgres**, or Grafana **Loki** (OSS) | |
| Load testing | **Locust** or **k6** (OSS) | How latency SLOs are validated for real |
| Compute | **Kaggle** (35 GPU-hr/wk), Colab free tier; AWS only for final serving demo | |

### 12.3 Methods (what production teams actually do)

- **Retrieval:** bi-encoder + ANN for recall, cross-encoder for precision (the two-tower → rerank pattern used by Google/Amazon-scale search); **hybrid search** (BM25 + dense, fused with Reciprocal Rank Fusion) — add via Milvus/Qdrant sparse support or `rank_bm25`, free and a known precision win.
- **Hard-negative mining** for embedding fine-tuning (mine negatives from your own ANN index — standard practice).
- **Quantization** (4-bit AWQ/GGUF) and **streaming generation** for latency; **semantic caching** of frequent queries (embed query → serve cached answer if cosine > threshold).
- **Test-time augmentation** and **confidence thresholding with human-in-the-loop routing** for the verification/counting path (how real fulfillment QA systems bound their error rates).
- **Shadow deployment / golden-set regression gates** in CI before promoting any model or prompt change.
- **Feature-consistent inference:** serialized preprocessing artifacts shared between training and serving (prevents train/serve skew — the #1 real-world ML bug).

### 12.4 Evaluation (industry-standard, all free)

| Layer | Metrics & tools |
|---|---|
| Retrieval | Recall@K, MRR, **NDCG@10** (the industry search metric) — via `pytrec_eval` or `ranx` (OSS) |
| Reranking | Precision@K uplift vs stage-1 only; ablation table |
| Detection/counting | mAP@50 (Ultralytics built-in), per-ASIN count accuracy, order exact-match rate |
| Verification | ROC-AUC, FAR/FRR, **calibration (ECE + reliability diagrams)** — scikit-learn |
| RAG/LLM quality | **Ragas** (faithfulness, answer relevance, context precision) + LLM-as-judge with **Promptfoo** for prompt regression testing — both OSS |
| Latency/SLO | P50/P95/P99 per stage under load (**Locust**), reported as an SLO table |
| Drift | PSI, KS-test, embedding-distribution monitoring (**Evidently**) |
| Human eval | Small blind side-by-side rating of responses (spreadsheet is fine) — teams always pair automated metrics with human review |

---

## 13. Cost Guardrails

- All training/experiments on Kaggle free GPU; AWS only for final serving.
- Single G4dn.xlarge (~$0.53/hr) — **stop when idle**; consider spot instances for non-demo hours.
- One-time captioning job; incremental indexing thereafter.
- Target total AWS spend: **< $60**.
