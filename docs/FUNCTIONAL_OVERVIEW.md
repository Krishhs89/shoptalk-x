# ShopTalk-X — Functional Overview

*What the system does, from a user's perspective. For "how it's built," see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).*

## What it is

A conversational shopping assistant over a ~10k-product catalog (Amazon
Berkeley Objects). You can search it three ways, and it can verify a
delivered item — both *which* item it is and *how many* — against what you
ordered.

## The five things a user can do

### 1. Ask in plain English
> "Show me a red shirt for men under 50 dollars"

The system finds relevant products and replies conversationally, citing
specific items by ID — not just a list of raw search results. Ask a
follow-up ("what about in blue?") and it remembers the conversation.

### 2. Search with a photo
Upload a picture of something you saw in person (or another product's
photo) and the system finds visually similar catalog items — same
conversational-answer treatment as text search.

### 3. Verify a delivery
Received a package and want to confirm it's actually what you ordered?
Upload a photo of the item + the order's product ID, and the system tells
you: **match**, **mismatch**, or **suspect** (uncertain — routed to a human
for review rather than guessing). This is the "did I get the wrong item /
a counterfeit" check.

### 4. Check a quantity
Claim you received the wrong count (e.g. "I ordered 3, only 2 arrived")?
Upload a photo and the claimed quantity, and the system counts visible
instances using a pretrained object detector and tells you: **match**,
**mismatch**, or **suspect** (close count, routed to human review) — or
**unsupported** if that product isn't a type the detector can recognize
(it only knows ~80 general object types, not this catalog's fine-grained
categories; see `docs/model_cards/quantity_counting.md`).

### 5. Give feedback
Thumbs up/down on any response. This is logged and, in a full production
loop, would feed the retraining pipeline (Day 6's Airflow DAG) so the
system improves from real usage over time.

## Where you interact with it

- **Chat UI** (Streamlit, `streamlit run src/shoptalk/ui/app.py`): the
  everyday interface — chat box, photo upload, conversation history, thumbs
  feedback, a "Verify order" tab, and a "Verify quantity" tab.
- **REST API directly** (`POST /search/text`, `POST /search/image`,
  `POST /verify`, `POST /verify/quantity`, `POST /feedback`): for
  programmatic access or building a different frontend. Full request/response shapes in
  `src/shoptalk/api/schemas.py`; interactive docs at `/docs` once the
  server is running (FastAPI auto-generates Swagger UI).
- **CLI** (`python -m shoptalk.retrieval.two_stage --query "..."`, etc.):
  every pipeline stage is also runnable standalone from the command line —
  useful for debugging or scripting.

## What happens behind a search, in plain terms

1. Your query (text or photo) gets compared against all ~10k products to
   find the 100 most plausibly relevant ones — fast, but approximate.
2. Those 100 get carefully re-ranked by a more accurate (but slower) model
   that looks at each candidate individually against your query — this is
   why the top few results are noticeably better than a single-pass search
   would give you.
3. The top handful of re-ranked products get handed to a language model,
   which writes the actual conversational answer, grounded in only those
   real products (it's instructed not to invent products or prices).

This two-step "fast-then-precise" retrieval pattern (and why it beats a
single-pass search) is explained with real measured numbers in
[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md#two-stage-retrieval).

## Honest caveats (things to know before you rely on this)

- **Prices are fake.** The underlying dataset (ABO) doesn't include prices.
  A price is synthesized per product (deterministic, category-appropriate
  range) purely so "under $50"-style queries are demoable. It's never real
  Amazon pricing — every product record flags this explicitly
  (`price_is_synthetic: true`).
- **This is a ~10k-product demo catalog**, not the full Amazon catalog —
  results are only as good as what's in that subset.
- **The verification model was trained at smoke-test scale** in this build
  session (183 example pairs). It works correctly end-to-end, but its
  confidence calibration will improve substantially once trained on the
  full catalog's much larger pool of examples — see
  [docs/model_cards/verification.md](model_cards/verification.md).
- **Response speed depends heavily on your hardware.** With a GPU (or Apple
  Silicon Metal acceleration working correctly), expect the sub-3-second
  responses the system is designed for. Without one, the language-model
  step in particular can be much slower — see the latency notes in
  [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md#latency--the-honest-numbers).

## What's built vs. documented-only vs. not attempted

| Capability | Status |
|---|---|
| Text search, image search, conversational answers | ✅ Built, tested with real data |
| Order verification (match/mismatch/suspect) | ✅ Built, tested with real data |
| Fine-tuned embeddings, MLflow tracking, drift detection | ✅ Built, tested with real data |
| Docker/docker-compose deployment | ✅ Built; validated locally |
| CI/CD (GitHub Actions) | ✅ Built; lint/test stage runs for real, cloud-deploy stages documented |
| AWS EC2 deployment | 📄 Documented step-by-step, not executed (see [why](deployment/aws_ec2.md)) |
| Airflow retraining DAG | ✅ Task logic validated directly; full Airflow orchestration documented, not run in this environment (no Docker available in the build sandbox) |
| LLM LoRA/QLoRA fine-tuning | 📄 Code + docs provided, not executed (per the problem statement's explicit allowance) |
| Personalization, voice input | ⚠️ Stretch goals — see the execution plan for status |
| Quantity validation (YOLO-based order-count check) | ❌ Not attempted — the largest remaining stretch item, needs an object-detection annotation pipeline beyond this session's scope |
| Demo video | ❌ Not possible for an AI agent to produce — you'll need to record this yourself following the UI walkthrough above |

## Where to go next

- Want to run it yourself? Start at the main [README.md](../README.md).
- Want the "why" behind a design choice? [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).
- Want per-model details (training data, eval numbers, failure modes)?
  [docs/model_cards/](model_cards/).
- Want the original assignment spec and grading rubric context?
  [docs/ShopTalk-X_Production_Design_Document.md](ShopTalk-X_Production_Design_Document.md).
