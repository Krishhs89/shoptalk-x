# ShopTalk-X — Session Log (2026-08-03)

Record of the Claude Code session covering initial document review and repo setup for the ShopTalk-X capstone project.

---

## 1. Document review request

**User:** "Please go through the documents for shoptalk project."

Claude read all four files in the project folder:

1. **`Copy of Problem Statement_ ShopTalk (1).pdf`** — the official course problem statement.
   - Core ask: text query in → relevant products + natural-language response (RAG + NLG).
   - Dataset: Amazon Berkeley Objects (ABO) — description, keywords, images (images captioned to augment text).
   - Core deliverables: preprocessing, embedding generation, vector DB (Chroma/Milvus), image captioning, LLM exploration, RAG pipeline (LangChain-friendly), Streamlit/Gradio UI, REST endpoint on AWS.
   - Optional: fine-tuned embeddings (triplet loss, LoRA/QLoRA), follow-up conversation support, personalisation, voice input, feedback loop.
   - Grading weights: EDA 15%, Model experimentation 25%, Deployment 25%, E2E testing 15%, UI/UX 5%, Documentation 15%.
   - FAQ explicitly scopes input/output as **text only**; image input is listed only under "Future Directions," not the core spec.

2. **`Elias Capstone projects 1 to 6 (1).pdf`** — a shared resource/links dump across all 6 capstone projects.
   - Contains links to the official ShopTalk problem statement + grading rubric (Google Docs/Sheets).
   - Resource list: image captioning models (CLIP/BLIP/ImageBind), `papermill` for notebook automation, LoRA/QLoRA tutorials, Ollama/Open WebUI, LangChain tutorials, RAG evaluation frameworks (Ragas/Promptfoo/Trulens), and BinSense-project resources (albumentations, SAM, LabelStudio) that the design doc later borrows from for verification/quantity-check features.

3. **`ShopTalk-X_Production_Design_Document.md`** — Krishna's own production design doc (v1.0, August 2026), read in full (13 sections):
   1. Problem Statement — discover (text/photo → products + LLM response) + verify (photo + order ID → match/mismatch/counterfeit check).
   2. Data — ABO subset (~10k products), BLIP captions, verification pairs (multi-view positives, ANN-mined hard negatives), albumentations robustness set.
   3. Model Architecture — bge/e5 bi-encoder fine-tuned w/ triplet loss + hard negatives; CLIP joint space; two-stage retrieval (ANN top-100 <50ms → cross-encoder rerank <300ms → LangChain RAG generation <2.5s via quantized Llama-3.1-8B); LoRA/QLoRA fine-tuning code deliverable; optional personalisation rerank blend; CV subsystems (YOLOv8 detect/crop, Siamese+MLP verification head, quantity validation, robustness augmentations).
   4. Inference Service & Latency — FastAPI, models loaded once at startup, per-stage latency budget table, optimization levers.
   5. Deployment — Docker/EC2 G4dn.xlarge, GitHub Actions CI/CD, MLflow registry promotion, Streamlit/Gradio UI w/ voice input.
   6. Monitoring — Prometheus/Grafana or CloudWatch, prediction logging, Evidently drift, Ragas LLM-quality checks, alerting.
   7. Continuous Improvement — feedback capture, golden eval set as CI regression gate, Airflow retraining DAG, incremental indexing, prompt versioning.
   8. Security & Governance — auth/rate-limiting, prompt-injection defenses, privacy/PII handling, model cards, auditability, bias/fairness slicing, human-in-the-loop for "suspect" verdicts.
   9. Evaluation Plan — metrics table per component.
   10. Milestones (6-week) + 10b Submission Checklist mapped to the official spec.
   11. Resume Bullets (draft, placeholder metrics).
   12. Industry-Standard Stack — specific free/OSS tool choices.
   13. Cost Guardrails — Kaggle for training, single EC2 instance stopped when idle, target <$60 AWS spend.

4. **`ShopTalk-X_7Day_Execution_Plan.md`** — compressed 7-day (~4hrs/day) build plan.
   - Scope decisions: KEEP two-stage retrieval, CLIP search, verification head, Docker+EC2, MLflow, Evidently, EDA, latency report, all docs. SIMPLIFIED Airflow (single local DAG) and monitoring (FastAPI /metrics + Evidently, skip Prometheus/Grafana). STRETCH (Day 7 only): quantity validation, voice input, personalisation. DROPPED: k3s/Kubernetes (document only), LLM LoRA run (code + docs only).
   - Day-by-day breakdown (Day 1 data/EDA/baseline retrieval → Day 7 docs/video/stretch), daily rhythm, and risk valves for common failure points.

**Claude's synthesis:** flagged that the design doc's scope (image search, verification, quantity validation) goes well beyond the official spec's core grade (which is text-in/text-out only) — those pieces map to "Future Directions" and the separate BinSense project, not the ShopTalk rubric itself. Noted as fine for ambition/stretch, just not core-graded.

---

## 2. Follow-up confirmation

**User:** "did you go through the productions design document?"

**Claude:** Confirmed yes, and gave a more detailed section-by-section recap of the Production Design Document (the 13-point breakdown above).

---

## 3. GitHub connection request

**User:** "please conect to https://github.com/Krishhs89/"

Claude checked `gh auth status` — already authenticated as `Krishhs89` (21 public repos at the time). Folder was not yet a git repository.

Asked two clarifying questions via AskUserQuestion (ambiguous "connect to", and repo naming/visibility):
- **What does "connect to" mean?** → User chose: **create a new repo & push this folder**.
- **Repo name/visibility?** → User chose: **`shoptalk-x` (private)**.

**Actions taken:**
1. `git init` in the Shoptalk project folder.
2. Staged and committed the 4 existing files (problem statement PDF, capstone links PDF, production design doc, 7-day execution plan) as the root commit.
3. `gh repo create Krishhs89/shoptalk-x --private --source=. --remote=origin --push` → created **https://github.com/Krishhs89/shoptalk-x** (private) and pushed `master`.

Result: local folder now tracks `origin` → `Krishhs89/shoptalk-x` on `master`.

---

## 4. This log

**User:** "please save the history from the begining of the conversation."

Claude wrote this file (`CONVERSATION_LOG.md`) summarizing the session from the first message onward, to be committed to the `shoptalk-x` repo alongside the project docs.
