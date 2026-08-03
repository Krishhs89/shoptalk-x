# Model Card: Generation LLM

## Intended use
Generates the final conversational, grounded response from the reranked
top-K products (design doc §3.2 "Stage 3 — Generation"). Constrained by
system prompt to answer only from retrieved products, cite `item_id`, and
decline out-of-domain questions (see `src/shoptalk/rag/prompts.py`).

## Model
`llama3.1:8b-instruct-q4_0` served locally via **Ollama**. 4-bit quantized
(per design doc's cost/latency guardrails). Not fine-tuned in this build —
LoRA/QLoRA instruction-tuning code + dataset-format spec is a documented,
runnable-on-Kaggle deliverable (`docs/finetuning/llm_lora_qlora.md`) rather
than an executed run, per the problem statement's explicit allowance ("LLM
LoRA run: deliver code + docs only").

## Prompting
- **Grounding**: system prompt requires citing retrieved `item_id`s; forbids
  inventing products/prices/claims not present in the `<catalog_results>`
  block.
- **Prompt-injection defense**: retrieved product text is wrapped in
  `<product item_id="...">` XML-ish delimiters with an explicit instruction
  that this content is DATA, not commands — blocks the pattern where a
  malicious product description contains "ignore previous instructions."
- **Conversation memory**: explicit turn history (not a LangChain Memory
  object — see `rag/chain.py`'s module docstring for why) capped at
  `llm.conversation_max_turns` (default 6) prior turns.

## Eval results
Validated end-to-end with real generations against real retrieved products
(see `results/day2_retrieval_eval.md` context + manual transcript review).
No formal Ragas/LLM-as-judge faithfulness scoring was run in this session
(design doc §6 lists it as a monitoring-stack component, not a one-time eval
gate) — `promptfoo`/`ragas` wiring is a natural Day-7+ extension once a
larger, real query volume exists to sample from.

## Latency caveat (important)
All latency numbers in `results/day6_locust*` were captured in a **CPU-only
sandbox with no GPU/Metal acceleration** — a single generation there took
30s–200s+ depending on concurrent load, vs. the design doc's <2.5s target on
real GPU hardware (Kaggle T4, EC2 G4dn.xlarge, or a Mac with working Metal
acceleration). This is a hardware constraint of the validation environment,
not a defect in the RAG chain or serving code — the same code, run on the
target hardware, is expected to meet the SLO. Re-run
`src/shoptalk/loadtest/locustfile.py` against your real deployment for
representative numbers.

## Limitations / failure modes
- **Grounding is prompt-enforced, not architecturally guaranteed**: an LLM
  can still hallucinate despite instructions. Production hardening would add
  a post-hoc citation check (does every `item_id` mentioned actually appear
  in the retrieved set?) before returning a response — not implemented here.
- **No content-safety filter beyond the system prompt** — appropriate for a
  product-search assistant on a curated catalog, but if this were extended
  to free-form chat, a dedicated moderation layer would be needed.
- **Small-model stand-ins during dev**: this session validated pipeline
  *correctness* using `qwen2.5:0.5b-instruct` as a fast stand-in where the
  8B model's CPU-only latency made iteration impractical. Response
  *quality* should be re-assessed against the real
  `llama3.1:8b-instruct-q4_0` default before treating any qualitative
  transcript from this session as representative.
