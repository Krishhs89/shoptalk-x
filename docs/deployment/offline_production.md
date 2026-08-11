# Offline Production Deployment (Local Docker)

This is the "offline" deployment target: the full stack running on a single
machine (a Mac, a Linux workstation, or any Docker host) with no cloud
dependency. It's the same images and compose files used on the AWS EC2
target in [`aws_ec2.md`](aws_ec2.md) — only *where* it runs differs. Use
this for local development, demos without internet-facing exposure, or as a
fallback if the cloud instance is stopped.

## 1. Prerequisites

- Docker + Docker Compose v2 (`docker compose version` should print `v2.x`).
- The processed catalog artifacts under `data/` — either run the pipeline
  yourself (see the README's "Full pipeline" section) or copy an existing
  `data/processed/`, `data/chroma/`, `data/models/bge-finetuned/`, and
  `data/verification/mlp_head.pt` from another machine.
- **macOS only**: Docker Desktop's default VM memory (often 2-5GB) is not
  enough to hold the API's loaded models (~2.7GB: bge, cross-encoder, CLIP,
  BLIP) plus an 8B q4 LLM (~5.3GB) at the same time — confirmed by hitting
  this for real (dropped connections, ~7x slower responses from swapping;
  see `results/day6_latency_report.md`). Docker Desktop → Settings →
  Resources → Memory → at least **12-14GB** before starting the stack.
- **Apple Silicon**: Docker Desktop can't pass through Metal, so `ollama`
  cannot use the GPU inside a container. Run `ollama serve` natively on the
  host instead (`brew install ollama`), and point `OLLAMA_BASE_URL` at
  `http://host.docker.internal:11434` in `docker-compose.yml`'s `api`
  service rather than the containerized `ollama:11434`.
- **Linux with an NVIDIA GPU**: install `nvidia-container-toolkit`, then
  uncomment the `deploy.resources.reservations` GPU block already present
  (commented) under the `ollama` service in `docker-compose.yml`, or apply
  the `docker-compose.prod.yml` overlay (see below) which does the same
  thing without editing the base file.

## 2. Bring the stack up

```bash
cd shoptalk-x

# fine-tuned embedding model is served by default (configs/config.yaml ->
# embeddings.text_model: "data/models/bge-finetuned") -- make sure that
# directory exists (either fine-tuned locally per the README, or the base
# BAAI/bge-base-en-v1.5 model will be used as a fallback if you never ran
# fine-tuning and skip this artifact)

docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_0
docker compose up -d --build api ui mlflow

# GPU host only (Linux + nvidia-container-toolkit):
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## 3. Require an API key (recommended even offline)

`SHOPTALK_API_KEY` is read from the environment by both `api` and `ui`
(`docker-compose.yml`'s `environment:` blocks). Set it before starting the
stack so `/search/*` and `/chat` reject unauthenticated requests:

```bash
export SHOPTALK_API_KEY="$(openssl rand -hex 24)"
docker compose up -d --build api ui
```

Without it set, `SHOPTALK_API_KEY` defaults to empty and the API runs
unauthenticated — fine for pure local dev, not for anything reachable by
anyone else on the network.

## 4. Smoke test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHOPTALK_API_KEY" \
  -d '{"query": "red shirt for men under 50 dollars"}'
```

Open `http://localhost:8501` for the UI, `http://localhost:5001` for
MLflow (experiment tracking + the model registry — see
`results/day5_finetune_eval.md` for the base-vs-fine-tuned comparison
logged there).

## 5. Retraining DAG (optional, same machine)

```bash
docker compose -f docker-compose.airflow.yml up -d
```

Airflow UI: `http://localhost:8080`. This runs the same
`shoptalk_x_retrain_embeddings` DAG documented in
[`aws_ec2.md`](aws_ec2.md)'s retraining section — pull data → rebuild
golden set → fine-tune → evaluate → promote (registers a new MLflow model
version only if Recall@10 improves) → trigger-deploy placeholder. See that
doc for the resource-tuning notes (`mem_limit`, `CUDA_VISIBLE_DEVICES`,
subprocess-vs-in-process execution) — they apply identically here; a
memory-constrained laptop is exactly the kind of host where the same OOM
failure mode would resurface if `mem_limit` and swap aren't sized
correctly.

## 6. Stopping / updating

```bash
docker compose down            # stop, keep volumes (models, chroma index, mlflow data)
docker compose pull            # if pulling prebuilt images from a registry
docker compose up -d --build   # rebuild from local source and restart
```

Data survives `down` because it's bind-mounted (`./data`, `./mlflow-data`)
or in a named volume (`ollama-models`) — only `docker compose down -v`
would delete it.

## 7. Health checks / autostart (optional)

For an "always-on" local deployment (e.g. a home server), add
`restart: unless-stopped` (already set on every service in
`docker-compose.yml`) and, on Linux, enable Docker's own systemd unit so
the daemon — and therefore the stack — comes back after a host reboot:

```bash
sudo systemctl enable docker
```

There is no process supervisor beyond Docker's own restart policy in this
setup (no k3s/systemd unit per service) — see `aws_ec2.md`'s "Kubernetes"
section for why that was scoped out.
