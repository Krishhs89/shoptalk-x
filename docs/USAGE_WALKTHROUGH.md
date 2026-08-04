# ShopTalk-X — Production Usage Walkthrough

*How to bring the system up and actually use it. Pairs with
[FUNCTIONAL_OVERVIEW.md](FUNCTIONAL_OVERVIEW.md) (what it does) and
[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) (how it's built).*

## Current deployment status

**Running locally right now** — the full stack (API + LLM + UI), built
against a real ~500-product slice of the ABO catalog, not the earlier
150-300-item smoke tests. This is the same code and same Docker images that
would run on AWS; only *where* it's hosted differs. AWS deployment is
pending real credentials for the account in play (see
`docs/deployment/aws_ec2.md` for the exact steps — nothing about the
process changes once those are available, it's a direct swap).

| Service | Local URL | What it is |
|---|---|---|
| **Chat UI** | http://localhost:8501 | The app — start here |
| API | http://localhost:8000 | REST backend |
| API docs (Swagger) | http://localhost:8000/docs | Interactive API explorer |
| MLflow | http://localhost:5000 | Experiment tracking / model registry |

**Important, measured live, not estimated:** on this Mac, Ollama is running
`llama3.1:8b-instruct-q4_0` **CPU-only** — no GPU/Metal acceleration is
available to it in this environment. A full `POST /search/text` request was
timed end-to-end: `stage1=224ms, rerank=1637ms, llm=1,008,506ms
(~16.8 min), total=1,010,367ms (~16.8 min)`. The answer itself was
accurate and well-grounded (correctly recommended a colorful phone case,
cited the right product ID and price, sensible ranked alternatives) — the
*quality* is real; the *hardware* is just far below what the design
targets. Retrieval + reranking together are under 2 seconds; the LLM step
alone accounts for 99.8% of total latency. A GPU (Kaggle T4, EC2
G4dn.xlarge) is expected to bring the LLM step to the design doc's <2.5s
target — see `results/day6_latency_report.md` for the full analysis and
`docs/model_cards/llm.md` for a faster stand-in model you can swap in
locally via `OLLAMA_MODEL=qwen2.5:0.5b-instruct` for responsive testing,
with the tradeoff that response *quality* will be lower than the real 8B
model. **Use the Verify tab and the API docs for a fast, live demo right
now; a chat-search turn with the full model takes roughly 15-20 minutes on
this machine until it's moved to GPU hardware.**

## Bringing it up yourself (if it's not already running)

```bash
cd Shoptalk
source .venv_prod/bin/activate    # or your own venv with `pip install -r requirements.txt`
export PYTHONPATH=src

# Terminal 1: LLM server
ollama serve
ollama pull llama3.1:8b-instruct-q4_0   # one-time

# Terminal 2: API
uvicorn shoptalk.api.main:app --host 0.0.0.0 --port 8000

# Terminal 3: UI
streamlit run src/shoptalk/ui/app.py
```

Or, once Docker is available: `docker compose up -d --build` brings up the
same four services in containers.

## Walkthrough: using the app

### 1. Open the chat UI
Go to **http://localhost:8501**. The sidebar shows API health and which
models are loaded — confirm it says "API online" before continuing.

### 2. Text search
Type a question in the chat box, e.g. *"show me a colorful phone case"* or
*"comfortable shoes under $50"* (remember: prices are synthetic — see the
functional overview). You'll get:
- A conversational answer citing specific product IDs
- A list of the retrieved products with category, price, and relevance score
- A latency footer (stage1 / rerank / LLM / total — this is the real
  per-stage breakdown described in `results/day6_latency_report.md`)

Ask a follow-up ("what about in black?") to see conversation memory in
action — it remembers the thread via a `session_id` carried across requests.

### 3. Photo search
Below the chat box, use the "Search by photo" uploader to pick an image
file. Submit with an empty text box (or type a note) — the system CLIP-
embeds your photo, finds visually similar catalog items, and BLIP-captions
your photo to ground the LLM's answer.

### 4. Voice search
Use the "Or ask by voice" recorder (same row as the photo uploader) to
speak your query instead of typing it. It transcribes locally via Whisper
and shows you the transcript before searching — correct it by just typing
instead if it mishears.

### 5. Give feedback
Every assistant response has 👍/👎 buttons. This logs to
`data/logs/predictions.db` and is the hook the retraining DAG
(`airflow/dags/retrain_embeddings_dag.py`) is designed to eventually close
the loop on.

### 6. Verify an order
Switch to the **"Verify order"** tab. Pick an item from the dropdown
(populated from the real catalog), upload a photo of the "received" item,
and click Verify. You'll get:
- 🟢 **match** — confidently the right item
- 🔴 **mismatch** — confidently the wrong item
- 🟡 **suspect** — ambiguous, routed to human review rather than an
  automatic accusation (see `docs/model_cards/verification.md`)

To see a clean **mismatch** demo, upload a photo of a *different* product
category than the one you selected — verified live: confidence 0.009 on a
threshold of 0.30, a confident correct rejection. The verification head was
trained on this deployment's actual 500-product catalog (378 pairs,
ROC-AUC 0.964, FAR 0.067, FRR 0.080 — see
`results/day5_verification_eval.md`), better-calibrated than the earlier
150-product smoke test but still smaller than the full ~10k-product target
(see the verification model card for how calibration is expected to keep
improving with scale).

### 7. Look under the hood (optional, for the technical audience)
- **http://localhost:8000/docs** — try any endpoint directly (e.g.
  `POST /search/text`) with real request/response schemas.
- **http://localhost:8000/metrics** — Prometheus-format counters/histograms
  per endpoint.
- **http://localhost:5000** — MLflow: browse the retrieval, fine-tuning, and
  verification experiment runs referenced throughout the model cards and
  results files.

## What data is actually behind this deployment

500 real English-language ABO products (86 categories) — larger than the
150-300-item samples used for component-by-component validation during
development, small compared to the ~10k-product target the design doc
specifies. Every part of the pipeline (download → preprocess → caption →
embed → index) that would run against the full 10k-product catalog ran here
against this 500-product slice — same code, smaller `--limit`. Scaling up is
a config change (`configs/config.yaml`'s `target_product_count`) plus
re-running the same commands with more time budgeted for captioning/
embedding.

## Moving this to AWS

Nothing about the app changes — `docs/deployment/aws_ec2.md` has the exact
EC2 launch, GPU-passthrough, and `docker compose up` steps. The practical
difference on a G4dn.xlarge: BLIP captioning and CLIP/text embedding run on
a real GPU instead of this Mac's CPU (minutes instead of the time this local
build took), and LLM generation drops from the tens-of-seconds-to-minutes
range seen in local CPU testing to the design doc's sub-3-second target.
