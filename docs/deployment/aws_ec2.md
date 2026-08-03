# AWS EC2 Deployment Guide

**Status: documented, not executed by Claude.** Per the design doc's cost
guardrails (§13) and the execution plan's Day-6 risk valve ("AWS friction →
local Docker demo + documented AWS steps"), provisioning and paying for real
AWS infrastructure is a decision for you to make and run yourself — an AI
agent should not spin up billable cloud infrastructure on your account
without you directly executing/approving each step. Everything below has
been validated *locally* (the same Docker images, the same docker-compose
stack); these steps only change *where* it runs.

## 1. Launch the instance

- Instance type: **`g4dn.xlarge`** (1x NVIDIA T4, 4 vCPU, 16GB RAM) — the
  target in design doc §5. On-demand ~$0.53/hr; use Spot for non-demo hours
  ([spot vs on-demand guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)).
- AMI: **Deep Learning AMI (Ubuntu)** — ships with NVIDIA drivers +
  `nvidia-container-toolkit` preinstalled, so Docker can see the GPU
  immediately.
- Security group: allow inbound `22` (SSH, restrict to your IP), `8000`
  (API), `8501` (UI), `5000` (MLflow) — do **not** open these to `0.0.0.0/0`
  in anything beyond a demo.
- Storage: 100GB gp3 (model weights + Docker images + catalog images add up
  fast — the Llama-3.1-8B q4 weights alone are ~4.7GB).

```bash
aws ec2 run-instances \
  --image-id <deep-learning-ami-id> \
  --instance-type g4dn.xlarge \
  --key-name <your-key-pair> \
  --security-group-ids <your-sg-id> \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=shoptalk-x}]'
```

## 2. First-time instance setup

```bash
ssh -i <your-key>.pem ubuntu@<instance-public-ip>

git clone https://github.com/Krishhs89/shoptalk-x.git
cd shoptalk-x
mkdir -p /opt/shoptalk-x && sudo mv * /opt/shoptalk-x/ && cd /opt/shoptalk-x

# verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 3. Build the catalog data on the instance (or `scp` it from your laptop)

The `data/processed/`, `data/chroma/`, and `data/verification/mlp_head.pt`
artifacts are NOT baked into the Docker image (see `.gitignore` — they're
either regenerable or too large for git). Either:

- **Regenerate on-instance** (slower, but no transfer needed): run the full
  pipeline per the README's "Run the pipeline" section, or
- **`scp` from your laptop** (faster if you already built it locally):
  ```bash
  scp -i <your-key>.pem -r data/processed data/chroma data/verification \
      ubuntu@<instance-public-ip>:/opt/shoptalk-x/data/
  ```

## 4. Enable GPU passthrough for Ollama and bring the stack up

Edit `docker-compose.yml` and uncomment the `deploy.resources.reservations`
block under the `ollama` service (already commented in, with instructions
inline), then:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_0
docker compose up -d --build api ui mlflow
```

## 5. Smoke test

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search/text -H "Content-Type: application/json" \
  -d '{"query": "red shirt for men under 50 dollars"}'
```

Open `http://<instance-public-ip>:8501` for the UI, `http://<instance-public-ip>:5000` for MLflow.

## 6. Cost control — stop when not actively demoing

```bash
aws ec2 stop-instances --instance-ids <instance-id>
# ... later ...
aws ec2 start-instances --instance-ids <instance-id>
```

A stopped instance still bills for its EBS volume (~$8/month at 100GB gp3)
but not compute. Terminate entirely (`aws ec2 terminate-instances`) once the
project is submitted if you don't need it anymore.

## 7. CI/CD hookup (optional)

`.github/workflows/ci.yml`'s `build-and-push` and `deploy` jobs are already
wired for this instance — they no-op safely if the relevant secrets aren't
set. To activate them, add these repo secrets (Settings → Secrets and
variables → Actions):

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM user scoped to ECR push only (least privilege) |
| `EC2_HOST` | instance public IP or DNS |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | private key matching the key pair above |

Once set, every push to `main` builds the API image, pushes it to ECR, and
does a rolling restart of the `api` container on the instance via SSH.

## Rollback

```bash
docker compose pull api   # or: docker pull <ecr-repo>:<previous-sha>
docker tag <ecr-repo>:<previous-sha> <ecr-repo>:latest
docker compose up -d api
```

## Kubernetes (documented, not executed — dropped per the 7-day scope decision)

The design doc lists EKS/k3s as a stretch orchestration target. Given the
scope decision to keep this a single-instance deployment (§ "Scope
decisions" in the execution plan), the concrete step-by-step for k3s is:

1. `curl -sfL https://get.k3s.io | sh -` on the EC2 instance (single-node
   k3s, no separate control plane needed for a demo).
2. Write a Deployment + Service manifest per container currently in
   `docker-compose.yml` (image references are unchanged — same images the
   CI pipeline already builds).
3. Add liveness/readiness probes pointing at `/health`.
4. `kubectl apply -f k8s/` and `kubectl port-forward` or a `LoadBalancer`
   Service to expose it.

Not executed here — it adds real operational complexity (probes, resource
limits, a second manifest set to keep in sync with docker-compose.yml) for a
single-instance demo that docker-compose already serves adequately.
