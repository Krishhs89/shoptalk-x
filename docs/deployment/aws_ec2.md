# AWS EC2 Deployment Guide (Online Production)

**Status: was live; currently unreachable due to lab-account access loss
(as of 2026-08-13).** This was executed for real, not just documented — a
`g4dn.xlarge` instance ran the full stack (API, UI, MLflow, Ollama) plus a
separate Airflow retraining stack, verified live end-to-end (see
[PROJECT_EXPLAINED.md](../PROJECT_EXPLAINED.md) for the evidence). As of
2026-08-13, the lab AWS account (`ik_mlsu_end_aug25cohort_g7`, a
time-boxed Nuvepro/cloudlabs training sandbox) started returning an
explicit IAM deny on `ec2:DescribeInstances`, and the instance itself
(`i-02615f77f0cbf82be`, last known IP `100.48.71.36`) stopped responding
to SSH entirely (connection timeout, not refused) — consistent with the
lab's access window expiring or the account being deprovisioned
server-side, not with anything wrong in this repo's code or
configuration. Re-checked as of 2026-08-16: no change, same two symptoms.

**This is not a regression to fix in code** — everything below remains
the accurate, tested procedure for standing the deployment back up. Once
fresh AWS credentials/access exist (either a renewed lab account or a
different AWS account), redeploying is: launch a same-spec instance,
follow steps 1-7 below verbatim, restore the swap file and `data/`/`results/`
permissions from §2a/§3a, and everything else — the git history, the
Docker images' build recipes, the fine-tuned model artifacts pattern —
is unchanged and ready to go. Everything below reflects what was actually
run, including the instance-level configuration (swap, file permissions)
that isn't captured anywhere in git because it lives on the host, not in
the repo.

## 1. Launch the instance

Actual instance used this deployment:

| | |
|---|---|
| Instance ID | `i-02615f77f0cbf82be` |
| Name tag | `shoptalk-x-gpu-pipeline` |
| Type | `g4dn.xlarge` (1x NVIDIA T4, 4 vCPU, 16GB RAM) |
| AMI | `ami-0b6f2229ad14c9323` (Deep Learning AMI, Ubuntu — ships with NVIDIA drivers + `nvidia-container-toolkit` preinstalled) |
| Region | `us-east-1` |
| Storage | 150GB gp3 (started at 90GB; grown once — see "Deploying a new dependency" below) |
| Key pair | `shoptalk-x-gpu` |
| Security group | `sg-00e8586a3ab415beb` |

Security group inbound rules actually applied:

| Port | Source | Purpose |
|---|---|---|
| 22 | your IP only (`x.x.x.x/32`) | SSH |
| 8000 | `0.0.0.0/0` | API — safe to open broadly because `SHOPTALK_API_KEY` is required (see step 5) |
| 8501 | `0.0.0.0/0` | UI |

MLflow (5001) and Airflow (8080) are **not** opened to the internet — they're
only reached via SSH tunnel or `curl localhost` on the instance itself. Keep
it that way; neither has its own auth layer.

```bash
aws ec2 run-instances \
  --image-id ami-0b6f2229ad14c9323 \
  --instance-type g4dn.xlarge \
  --key-name shoptalk-x-gpu \
  --security-group-ids sg-00e8586a3ab415beb \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=90,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=shoptalk-x-gpu-pipeline}]'
```

IAM note: the lab-account credentials used for this deployment deny
`sagemaker:*` and Spot requests, and are region-locked to `us-east-1` — plan
around EC2 on-demand + ECR/S3 only if you're on a similarly restricted
account.

## 2. First-time instance setup

```bash
ssh -i shoptalk-x-gpu.pem ubuntu@<instance-public-ip>

cd ~   # /home/ubuntu -- `git clone` below creates /home/ubuntu/shoptalk-x directly
git clone https://github.com/Krishhs89/shoptalk-x.git
cd shoptalk-x

# verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Cloning a private repo needs a PAT in the URL
(`https://<token>@github.com/...`) or an SSH deploy key — either way,
**strip the token from the git remote after cloning** so it doesn't sit in
plaintext in `.git/config` on a machine other people might access:

```bash
git remote set-url origin https://github.com/Krishhs89/shoptalk-x.git
history -c   # clear shell history if the PAT was typed on the command line
```

### 2a. Add swap (required — 16GB RAM is not enough headroom)

This instance runs the serving stack (API with 4 loaded models + Ollama's
8B LLM) AND, at times, the Airflow retraining stack concurrently. Real
memory exhaustion was hit during retraining DAG runs (container RSS >11GB,
system free memory dropped to ~227MB, `kswapd0` pegged at 60%+ CPU) with no
swap configured — Linux OOM behavior under that condition is unpredictable
(anything from a killed container to the instance shutting down outright,
both of which happened here during debugging). Set up an 8GB swap file
before running anything heavy:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # persist across reboots
free -h   # confirm Swap: 8.0Gi
```

## 3. Build the catalog data on the instance (or `scp` it from your laptop)

`data/processed/`, `data/chroma/`, `data/models/bge-finetuned/`, and
`data/verification/mlp_head.pt` are not in git (see `.gitignore`). Either
regenerate on-instance (slower) or copy from a machine that already has
them:

```bash
scp -i shoptalk-x-gpu.pem -r data/processed data/chroma data/models data/verification \
    ubuntu@<instance-public-ip>:/home/ubuntu/shoptalk-x/data/
```

### 3a. File permissions — needed before Airflow can write here

If you also run the Airflow retraining stack (step 7) against this same
checkout, its container runs as `uid=50000 gid=0` while directly-`scp`'d or
`git clone`'d files are typically owned by `ubuntu:ubuntu` with no
group/other write bit. This bit twice, in two different directories, both
*after* the expensive part had already finished — the DAG's compute (13
minutes of fine-tuning, ~70 minutes of full-catalog re-encoding for
evaluation) completed successfully both times, only to fail writing the
output: first `PermissionError: ... 'data/models/bge-finetuned/
config_sentence_transformers.json'` (saving the new model), then the same
error on `results/day5_finetune_eval.md` (writing the eval report) once
that first one was fixed. Pre-empt both at once, on both directories the
DAG writes to:

```bash
sudo chmod -R a+rwX /home/ubuntu/shoptalk-x/data /home/ubuntu/shoptalk-x/results
```

## 4. Enable GPU passthrough and bring the serving stack up

`docker-compose.prod.yml` is a compose overlay (not an edit to the base
file) that adds the `nvidia` GPU reservation to `ollama`:

```bash
cd /home/ubuntu/shoptalk-x
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d ollama
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_0
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build api ui mlflow
```

The API serves the **fine-tuned** embedding model by default
(`configs/config.yaml` → `embeddings.text_model:
"data/models/bge-finetuned"`), with `embeddings.base_text_model:
"BAAI/bge-base-en-v1.5"` kept separately as the pretrained starting point
for future fine-tuning runs and the evaluation floor — the serving model
and the "base" comparison model are deliberately different config keys so
retraining never accidentally fine-tunes an already-fine-tuned model.

## 5. Require an API key

```bash
export SHOPTALK_API_KEY="$(openssl rand -hex 24)"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build api ui
```

With port 8000 open to `0.0.0.0/0`, this is not optional in this
deployment — every `/search/*` and `/chat` call must include
`X-API-Key: $SHOPTALK_API_KEY`.

## 6. Smoke test

```bash
curl http://<instance-public-ip>:8000/health

curl -X POST http://<instance-public-ip>:8000/search/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHOPTALK_API_KEY" \
  -d '{"query": "red shirt for men under 50 dollars"}'
```

UI: `http://<instance-public-ip>:8501`. MLflow (SSH-tunnel only —
`ssh -i shoptalk-x-gpu.pem -L 5001:localhost:5001 ubuntu@<ip>`, then
`http://localhost:5001`).

## 7. Retraining DAG (Airflow, same instance)

```bash
docker compose -f docker-compose.airflow.yml up -d
```

`docker-compose.airflow.yml` settings that matter here and why:

```yaml
environment:
  MLFLOW_TRACKING_URI: "http://host.docker.internal:5001"   # Linux needs the extra_hosts entry below to resolve this
  CUDA_VISIBLE_DEVICES: ""    # keep training off the GPU -- it's reserved for Ollama/serving; avoided a real contention hang
  OMP_NUM_THREADS: "2"
  MKL_NUM_THREADS: "2"
extra_hosts:
  - "host.docker.internal:host-gateway"   # host.docker.internal only resolves on Docker Desktop by default; this makes it work on Linux too
mem_limit: "10g"    # cgroup cap -- see below for why this number specifically
```

`mem_limit: 10g` was tuned empirically, not guessed: 6GB caused a clean
SIGKILL of the fine-tuning task (`exit code -9`) partway through; 10GB let
it complete. The actual constraint that mattered more than the limit
itself, though, was *where the DAG's Python callables execute their heavy
work*: importing `SentenceTransformer` and calling `finetune_main()`
directly inside Airflow's long-lived `PythonOperator` process showed
genuine unbounded memory growth within a single training epoch (7 tiny
steps), while the identical work run as a `python -m module` **subprocess**
completed in under 2 minutes with no such growth. `airflow/dags/
retrain_embeddings_dag.py`'s `task_finetune_embeddings` and
`task_evaluate_and_promote` both shell out via a `_run_subprocess()` helper
for exactly this reason — don't revert that to an in-process call.

**Before doing any DAG debugging, pause it first:**

```bash
docker exec shoptalk-x-airflow-1 airflow dags pause shoptalk_x_retrain_embeddings
```

The default `retries: 1` / `retry_delay: 5 minutes` in `default_args`
means an in-progress fix that isn't deployed within 5 minutes of a failure
gets raced by an automatic retry using the *old* (still-broken) code —
this happened once here and the resulting memory exhaustion escalated all
the way to an actual instance shutdown (`Client.UserInitiatedShutdown`),
which needed `aws ec2 start-instances` (not just a reboot) to recover, and
changed the instance's public IP in the process. `airflow dags unpause`
once the fix is deployed and you're ready to trigger a fresh run.

Airflow UI is reachable only via SSH tunnel
(`ssh -i shoptalk-x-gpu.pem -L 8080:localhost:8080 ubuntu@<ip>`) →
`http://localhost:8080`, or via `docker exec ... airflow tasks
states-for-dag-run <dag_id> <run_id>` for a quick CLI status check without
a tunnel.

## 8. Cost control — stop when not actively demoing

```bash
aws ec2 stop-instances --instance-ids i-02615f77f0cbf82be
# ... later ...
aws ec2 start-instances --instance-ids i-02615f77f0cbf82be
```

**Stopping changes the public IP** (this happened during this deployment:
`34.229.65.25` → `100.48.71.36` after a stop/start cycle) — re-check
`describe-instances` and update any bookmarked URL/security-group source
IP after restarting. A **reboot** (`aws ec2 reboot-instances`) keeps the
same IP if you just need to restart the OS without the IP changing.

A stopped instance still bills for its 150GB gp3 EBS volume (~$12/month) but
not compute. Terminate (`aws ec2 terminate-instances`) once you no longer
need it.

## 9. CI/CD hookup (optional)

`.github/workflows/ci.yml`'s `build-and-push` and `deploy` jobs no-op
safely if the relevant secrets aren't set. To activate:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | IAM user scoped to ECR push only |
| `EC2_HOST` | instance public IP or DNS |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | private key matching `shoptalk-x-gpu` |

Once set, every push to `main` builds the API image, pushes it to ECR, and
does a rolling restart of the `api` container via SSH.

## Rollback

```bash
docker compose pull api   # or: docker pull <ecr-repo>:<previous-sha>
docker tag <ecr-repo>:<previous-sha> <ecr-repo>:latest
docker compose up -d api
```

## Deploying a new dependency (e.g. the quantity-validation feature)

Adding `ultralytics` (for `shoptalk.counting.*`) to `requirements/serving.txt`
and rebuilding the API image on this instance surfaced two real,
instance-specific problems worth knowing about before doing this again:

- **The 90GB root EBS volume is too small once a heavy new dependency is
  added.** `ultralytics` pulls in a full CUDA 13 toolkit via `torch`'s
  split-wheel packaging (`nvidia-cublas`, `nvidia-cudnn`, etc. — several GB
  of wheels), and the build failed twice with `no space left on device`,
  first during the pip install's temp files, then again extracting a
  layer. The instance also has a large second NVMe volume (~116GB, free
  with the `g4dn.xlarge`, mounted at `/opt/dlami/nvme`) — resist the
  temptation to just relocate Docker's data-root there instead of growing
  the EBS volume: that volume is **ephemeral instance store**, wiped on
  every stop/start (not reboot), and this doc's own §8 cost-control
  workflow *is* a stop/start cycle — you'd silently lose every image on
  the next "stop when not demoing." Grow the persistent EBS volume instead:
  ```bash
  aws ec2 modify-volume --volume-id <vol-id> --size 150   # gp3, ~$0.08/GB-month, trivial cost
  # wait for ModificationState to reach "optimizing" (usable immediately, background-optimizes further)
  sudo growpart /dev/nvme0n1 1     # confirm device/partition via lsblk first
  sudo resize2fs /dev/nvme0n1p1
  ```
- **`ultralytics` needs `opencv-python` (not headless), which needs system
  shared libs the slim base image doesn't have** — surfaces as
  `ImportError: libxcb.so.1: cannot open shared object file`, and only on
  the *first actual call* that imports it (the import is deliberately
  deferred in `count.py`), not at container startup or in `/health`. The
  Dockerfile now installs `libgl1 libglib2.0-0 libxcb1 libxrender1
  libxext6 libsm6` in the runtime stage for this reason — don't remove
  them.
- **A freshly-retrained model checkpoint can be unreadable by the serving
  container even after the `data/` chmod fix in §3a** — that chmod only
  fixes files that exist *at the time you run it*. A file the Airflow
  container creates *afterward* (e.g. a new `model.safetensors` from a
  later retrain) gets that container's own restrictive default
  permissions (`rw-------`, owner-only). This surfaced as
  `FileNotFoundError: model.safetensors` on the API's *next restart*
  (not immediately — a container that already had the model loaded in
  memory before the retrain never re-reads the file, so the break is
  invisible until something restarts). Re-run the §3a chmod after every
  retrain, or expect to hit this again.

## Operational lessons (read before debugging a hang on this instance)

- **When SSH itself stops responding**, don't just keep retrying SSH — the
  instance may be genuinely unresponsive (not just slow) while AWS's own
  system-status check still reports "ok" (that check only covers the
  hypervisor, not guest-OS responsiveness; "instance status: impaired" is
  the signal that actually means "SSHD isn't answering"). Write resource
  diagnostics to a **local file on the instance** (`vmstat`, `free -h`,
  `docker stats --no-stream`, on a `while true; sleep 10` loop, `nohup`'d
  and disowned) *before* you need them — that log survives an SSH hang and
  is often the only way to find the real root cause after the fact, versus
  guessing.
- **GPU contention was the first (wrong) hypothesis** for the hangs seen
  here — it looked plausible (LLM + training both wanting the GPU) but
  `CUDA_VISIBLE_DEVICES=""` alone did not fix it. The real cause was memory
  exhaustion, then (after that was fixed) an in-process execution memory
  leak. Don't stop at the first plausible theory — confirm with a metric
  (`torch.cuda.is_available()`, `docker stats`, `free -h`) before declaring
  a fix.
- **`subprocess.run(cmd, check=True)` without `capture_output=True` is
  invisible in Airflow's task log** — Airflow's log redirect only
  intercepts the *current* process's `sys.stdout`, not a child process's OS
  file descriptors. If a subprocess-based task fails with no useful detail
  in the log, this is almost certainly why; capture and `print()` the
  output explicitly.

## Kubernetes (documented, not executed — dropped per the 7-day scope decision)

The design doc lists EKS/k3s as a stretch orchestration target. Given the
scope decision to keep this a single-instance deployment, the concrete
step-by-step for k3s would be:

1. `curl -sfL https://get.k3s.io | sh -` on the EC2 instance (single-node
   k3s, no separate control plane needed for a demo).
2. Write a Deployment + Service manifest per container currently in
   `docker-compose.yml` (same images the CI pipeline already builds).
3. Add liveness/readiness probes pointing at `/health`.
4. `kubectl apply -f k8s/` and `kubectl port-forward` or a `LoadBalancer`
   Service to expose it.

Not executed — it adds real operational complexity (probes, resource
limits, a second manifest set to keep in sync with docker-compose.yml) for
a single-instance demo that docker-compose already serves adequately.
