# ShopTalk-X — Workflows

*Visual walkthroughs of how a request moves through the system, and how the
system itself gets built, deployed, and kept up to date. Pairs with
[USAGE_WALKTHROUGH.md](USAGE_WALKTHROUGH.md) (how to click through the app)
and [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) (component map,
design decisions).*

## 1. Text search — request workflow

What happens between you typing a question and getting an answer.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant API as FastAPI (/search/text)
    participant S1 as Stage-1 ANN (bge + Chroma)
    participant RR as Cross-encoder rerank
    participant LLM as Ollama (Llama-3.1-8B)
    participant DB as SQLite prediction log

    User->>UI: types a question
    UI->>API: POST /search/text {query, session_id}
    API->>S1: embed query, ANN search (top-100)
    S1-->>API: 100 candidates (~200ms)
    API->>RR: rerank(query, candidates)
    RR-->>API: top-K reordered (~1.6s)
    API->>LLM: prompt (system + catalog_block + history + query)
    LLM-->>API: streamed/complete grounded answer
    API->>DB: log query, retrieved IDs, scores, latency
    API-->>UI: {answer, hits, latency, session_id, request_id}
    UI-->>User: renders answer + product cards + latency footer
    User->>UI: 👍/👎 (optional)
    UI->>API: POST /feedback {request_id, rating}
    API->>DB: log feedback
```

**Measured, real numbers for this exact flow** (500-product local
deployment, CPU-only): stage-1 ~224ms, rerank ~1,637ms, LLM ~16.8 min (see
`results/day6_latency_report.md` for the full breakdown and why the LLM
step dominates on this hardware).

## 2. Photo search — request workflow

Same shape, different stage-1/rerank inputs.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant API as FastAPI (/search/image)
    participant CLIP as CLIP ANN (OpenCLIP + Chroma)
    participant BLIP as BLIP captioner
    participant RR as Cross-encoder rerank
    participant LLM as Ollama

    User->>UI: uploads a photo
    UI->>API: POST /search/image {file}
    API->>CLIP: embed photo, ANN search (top-50)
    CLIP-->>API: 50 visually-similar candidates
    API->>BLIP: caption the QUERY photo itself
    BLIP-->>API: pseudo-text query, e.g. "a pair of brown suede shoes"
    API->>RR: rerank(pseudo_query, candidates)
    RR-->>API: top-K reordered
    API->>LLM: prompt referencing the pseudo-query + reranked products
    LLM-->>API: grounded answer
    API-->>UI: {answer, hits, pseudo_query, latency}
```

The reranker never sees the photo — only the BLIP-generated caption. This
is why photo-search quality depends on caption accuracy (see
`docs/model_cards/clip_and_captioning.md`).

## 3. Order verification — request workflow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI (Verify tab)
    participant API as FastAPI (/verify)
    participant CLIP as CLIP embedder
    participant MLP as Trained verification head

    User->>UI: selects ordered item_id, uploads received-item photo
    UI->>API: POST /verify?order_item_id=... {file}
    API->>CLIP: embed received photo + catalog photo
    CLIP-->>API: two embedding vectors
    API->>MLP: pair_features(emb_a, emb_b) -> sigmoid
    MLP-->>API: confidence score
    API->>API: compare to threshold +/- suspect margin
    API-->>UI: {verdict: match|mismatch|suspect, confidence, threshold}
    UI-->>User: 🟢/🔴/🟡 badge + explanation
```

`suspect` never auto-accuses — it's a deliberate band around the trained
threshold that routes ambiguous cases to a human (design doc §8).

## 4. Data + model pipeline — build workflow

What "building the catalog" actually runs, in order, once.

```mermaid
flowchart LR
    A[download_abo.py<br/>pull ABO subset from S3] --> B[preprocess.py<br/>clean + join fields]
    B --> C[caption_images.py<br/>BLIP captions -> document field]
    C --> D1[embed_text.py<br/>bge -> Chroma text collection]
    C --> D2[embed_image.py<br/>CLIP -> Chroma image collection]
    D1 --> E[generate_golden_set.py<br/>query -> relevant-products test set]
    E --> F[evaluate_retrieval.py<br/>Recall/MRR/NDCG, stage1 vs two-stage]
    D1 --> G[finetune_text.py<br/>triplet loss + hard negatives]
    G --> H[eval_finetune.py<br/>base vs fine-tuned Recall@K]
    B --> I[build_pairs.py<br/>Siamese pairs, multi-view + hard negatives]
    I --> J[train_verification.py<br/>MLP head, ROC-AUC/FAR/FRR]
    E --> K[finetune LLM: prepare_dataset.py<br/>+ train_lora.py, Kaggle GPU]
```

Every box is a real, independently runnable `python -m shoptalk....` command
(see the README's "Full pipeline" section for the exact commands and order).

## 5. CI/CD + retraining — operational workflow

```mermaid
flowchart TD
    subgraph CI["On every push (.github/workflows/ci.yml)"]
        direction LR
        L[lint: ruff] --> T[test: pytest, 29 tests]
        T --> B2{On main branch<br/>+ AWS secrets set?}
        B2 -->|yes| BD[build + push image to ECR]
        BD --> DP[deploy: SSH + docker compose up on EC2]
        B2 -->|no| SK[skip cloud steps cleanly]
    end

    subgraph Retrain["Airflow DAG (scheduled weekly or drift-triggered)"]
        direction LR
        RD[pull data] --> RB[rebuild golden set /<br/>hard-neg triplets]
        RB --> RF[fine-tune embeddings]
        RF --> RE[evaluate vs golden set]
        RE --> RG{Beats current<br/>model?}
        RG -->|yes| RP[register in MLflow,<br/>trigger deploy]
        RG -->|no| RN[log + discard,<br/>no promotion]
    end

    subgraph Monitor["Ongoing"]
        M1[every request logged<br/>to SQLite] --> M2[Evidently drift report<br/>on query distribution]
        M2 -->|drift detected| Retrain
    end
```

The retraining DAG's promotion gate was validated directly against real
trained models (see `airflow/dags/retrain_embeddings_dag.py`'s
`task_evaluate_and_promote` and the commit history) — it only promotes a
new model version to the MLflow registry if it actually beats the current
one on the golden set, never blindly.

## Where each workflow's code lives

| Workflow | Entry point(s) |
|---|---|
| Text search | `shoptalk/api/main.py::search_text`, `shoptalk/rag/chain.py` |
| Photo search | `shoptalk/api/main.py::search_image`, `shoptalk/retrieval/image_search.py` |
| Verification | `shoptalk/api/main.py::verify`, `shoptalk/verification/verify.py` |
| Data/model pipeline | `shoptalk/data/`, `shoptalk/embeddings/`, `shoptalk/eval/`, `shoptalk/verification/` |
| CI/CD | `.github/workflows/ci.yml` |
| Retraining | `airflow/dags/retrain_embeddings_dag.py` |
| Monitoring | `shoptalk/monitoring/drift_report.py`, `shoptalk/api/logging_store.py` |
