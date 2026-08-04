# ShopTalk-X — Production Usage Walkthrough

*How to bring the system up and actually use it. Pairs with
[FUNCTIONAL_OVERVIEW.md](FUNCTIONAL_OVERVIEW.md) (what it does) and
[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) (how it's built).*

## Current deployment status

**Running locally right now, in Docker** — `docker compose up -d` brings up
all four services as real containers (`restart: unless-stopped`, so they
survive crashes and, per the note in the previous conversation on process
ancestry, VS Code closing), built against a real ~500-product slice of the
ABO catalog. This is the exact same images/config that would run on AWS;
only *where* it's hosted differs. AWS deployment is pending real credentials
for the account in play (see `docs/deployment/aws_ec2.md` — nothing about
the process changes once those are available, it's a direct swap).

| Service | Local URL | What it is |
|---|---|---|
| **Chat UI** | http://localhost:8501 | The app — start here |
| API | http://localhost:8000 | REST backend |
| API docs (Swagger) | http://localhost:8000/docs | Interactive API explorer |
| MLflow | http://localhost:5001 | Experiment tracking / model registry (not 5000 -- macOS's AirPlay Receiver squats on 5000 by default) |

### Docker was validated for real, and found (+ fixed) three real bugs

`docker compose up --build` had never actually been run end-to-end before
this pass — the configs were written and reasoned through, not executed.
Running it for real surfaced:
1. **UI image build failure** — `docker-compose.yml`'s `ui` service set the
   build *context* to `./docker/ui`, but its Dockerfile does `COPY src/
   src/` / `COPY configs/ configs/`, which only exist at the repo root.
   Fixed: context now `.`, dockerfile explicitly `docker/ui/Dockerfile`.
2. **API image build failure** — the root `Dockerfile` only copied
   `requirements.txt` (which, since the Day-7 per-component split, is just
   `-r requirements/*.txt` references), not the `requirements/` directory
   those references point at. Fixed: copies `requirements/` and installs
   `requirements/serving.txt` directly (leaner image, skips UI/EDA/dev-only
   deps this service never imports).
3. **Every containerized request 401'd** — `docker-compose.yml` passes
   `SHOPTALK_API_KEY: "${SHOPTALK_API_KEY:-}"`, which sets an **empty
   string**, not "unset", when the host has no such variable. The API's
   `if API_KEY is None: disable auth` check doesn't catch `""`, so it
   silently enforced an empty-string key nothing could match. Fixed
   (`security.py`: `os.environ.get(...) or None`), with regression tests
   covering unset / empty / real-key cases.
4. **`/verify` 500'd inside the container** — `products.parquet`'s
   `image_path` column stores an *absolute host path*, baked in wherever
   `preprocess.py` last ran. The container mounts the same files at a
   different absolute path (`/app/data/...` vs the host's
   `/Users/.../Shoptalk/data/...`), so the stored path didn't resolve.
   Fixed: `verify.py` now reconstructs the catalog image path from
   `image_id` + the *current* process's `raw_dir` config at request time,
   rather than trusting a path baked in by a possibly different
   environment (also relevant for the Colab/Kaggle-built-then-locally-
   served workflow, not just Docker).

All four fixed and re-verified live: `/health`, `/verify` (6.4s, correctly
flagged a real mismatch), and a full `/search/text` round trip through the
container network (retrieval+rerank 8.5s, LLM 24.7s with the fast stand-in
model, correct grounded answer) all passed post-fix.

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

The UI's own request timeout is set generously (`SHOPTALK_SEARCH_TIMEOUT_S`,
default 1800s = 30 min) specifically so it doesn't abort a slow-but-working
CPU-only LLM call with a `Read timed out` error -- if you push generation
past 30 minutes (e.g. a much bigger prompt or an even slower machine), raise
that env var before starting the UI: `SHOPTALK_SEARCH_TIMEOUT_S=3600
streamlit run src/shoptalk/ui/app.py`.

## Bringing it up yourself (if it's not already running)

**Docker (recommended — this is what's actually running):**
```bash
open -a Docker                       # start Docker Desktop, wait for it to be ready
cd Shoptalk
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_0   # one-time, ~4.7GB
```
`docker compose ps` to check status; `docker compose logs -f api` to tail
logs; `docker compose down` to stop everything (`-v` to also wipe the
Ollama model volume).

**Native (no Docker), if you'd rather not run the daemon:**
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
Note: native `ollama serve` and the Docker `ollama` container both want
port 11434 — stop one before starting the other (`pkill -f "ollama serve"`
or `docker compose stop ollama`).

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
- **http://localhost:5001** — MLflow: browse the retrieval, fine-tuning, and
  verification experiment runs referenced throughout the model cards and
  results files.

## What data is actually behind this deployment

500 real English-language ABO products (86 categories) — larger than the
150-300-item samples used for component-by-component validation during
development, small compared to the ~10k-product target the design doc
specifies. Every part of the pipeline (download → preprocess → caption →
embed → index) that would run against the full 10k-product catalog ran here
against this 500-product slice — same code, smaller `--limit`.

## Scaling up on a free GPU (Colab / Kaggle), before AWS is ready

`notebooks/03_gpu_pipeline_colab_kaggle.ipynb` runs the same pipeline (full
~10k-product scope by default) on a free Colab GPU (Google One's bundled
Colab compute works) or Kaggle's free GPU quota — captioning, embedding, and
fine-tuning are all dramatically faster there than on this Mac's CPU. It
zips the resulting `data/` + `results/` artifacts for download; the
notebook's last section has the exact `rsync` commands to merge them into
this local deployment and restart the API to pick them up. Colab/Kaggle are
for this batch step only, not for hosting the live service (session limits,
no stable public URL) — see `docs/deployment/aws_ec2.md` for that.

## Moving this to AWS

Nothing about the app changes — `docs/deployment/aws_ec2.md` has the exact
EC2 launch, GPU-passthrough, and `docker compose up` steps. The practical
difference on a G4dn.xlarge: BLIP captioning and CLIP/text embedding run on
a real GPU instead of this Mac's CPU (minutes instead of the time this local
build took), and LLM generation drops from the tens-of-seconds-to-minutes
range seen in local CPU testing to the design doc's sub-3-second target.
