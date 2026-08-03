# LLM Instruction Fine-Tuning: LoRA / QLoRA

**Status: code delivered and runnable; training run not executed.** Per the
problem statement's explicit allowance ("LLM LoRA run: deliver code + docs
only") and because full/QLoRA fine-tuning of an 8B model needs a real GPU
with 12GB+ VRAM (this project's CPU-only validation sandbox has none). The
code in `src/shoptalk/llm_finetune/` is complete and meant to run as-is on
Kaggle's free GPU tier — not a stub.

## What was actually run

`prepare_dataset.py` **was executed for real** against the live pipeline:
it runs each golden-set query through the actual production two-stage
retriever, then has the actual production LLM chain write the target
response — so the resulting `data/llm_finetune/instructions.jsonl` reflects
genuine retrieval + generation behavior, not fabricated examples.

## Dataset format

Each line of `data/llm_finetune/instructions.jsonl`:
```json
{
  "instruction": "You are ShopTalk, a shopping assistant. Given the user's question and the retrieved catalog products below, answer helpfully and cite product IDs.",
  "query": "red shirt for men under 50 dollars",
  "retrieved_context": "<product item_id=\"...\">...</product>\n<product item_id=\"...\">...</product>",
  "response": "I found a couple of options... (id: B0XXXXX)"
}
```

`train_lora.py` formats each row into Llama-3's chat template
(`<|start_header_id|>system/user/assistant<|end_header_id|>...`), with the
`response` field as the training target — the model learns to reproduce
grounded, cited answers from `(instruction, query, retrieved_context)`.

## Running it (on Kaggle GPU or similar)

```bash
pip install -r requirements/llm_finetune.txt

# 1. Generate more training examples once you have the full golden set
python -m shoptalk.llm_finetune.prepare_dataset --n 100

# 2a. Standard LoRA (needs ~16GB+ VRAM for an 8B model in bf16)
python -m shoptalk.llm_finetune.train_lora \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output-dir data/models/llama31-8b-lora

# 2b. QLoRA (4-bit quantized base + LoRA adapters — fits Kaggle's free T4 16GB)
python -m shoptalk.llm_finetune.train_lora \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output-dir data/models/llama31-8b-qlora \
  --qlora
```

## LoRA vs. QLoRA

| Criteria | LoRA | QLoRA |
|---|---|---|
| Definition | Fine-tune by learning low-rank updates | LoRA + quantization of the frozen base weights |
| Efficiency | Reduces number of trained parameters | Further reduces memory & compute via quantization |
| Memory usage | Moderate (~16GB+ for 8B in bf16) | Very low (~6-8GB, fits a free-tier T4) |
| Performance hit | Minimal | Slightly lower; more aggressive quantization = larger hit |
| Hardware target | Resource-limited (single mid-tier GPU) | Highly resource-constrained (Kaggle/Colab free tier) |

Given the project's cost guardrails (§13: Kaggle free GPU-hours only), QLoRA
(`--qlora`) is the practical default — it's what actually fits a free
Kaggle session.

## Config choices and why

- `r=16, lora_alpha=32` — a common, well-tested starting ratio (alpha = 2×r)
  for instruction fine-tuning; increase `r` if the adapter underfits on a
  larger dataset.
- `target_modules` covers both attention projections (`q/k/v/o_proj`) and
  MLP projections (`gate/up/down_proj`) — attention-only LoRA is cheaper but
  MLP layers carry meaningful capacity for this kind of stylistic/grounding
  adaptation.
- `bnb_4bit_quant_type="nf4"` — the quantization format QLoRA's paper found
  best-preserves quality vs. plain int4.
- `report_to="mlflow"` — training runs land in the same MLflow tracking
  server as every other experiment in this project (`configs/config.yaml`'s
  `mlflow.tracking_uri`), so a fine-tuned LLM run sits alongside the
  embedding/verification runs for comparison.

## Merge and serve steps

After training, merge the LoRA adapter into the base weights for serving
(Ollama serves plain GGUF, not raw PEFT adapters):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
merged = PeftModel.from_pretrained(base, "data/models/llama31-8b-qlora").merge_and_unload()
merged.save_pretrained("data/models/llama31-8b-merged")
```

Then convert to GGUF for Ollama (using `llama.cpp`'s `convert_hf_to_gguf.py`,
quantize to your target bit-depth, write an Ollama `Modelfile` pointing at
the resulting `.gguf`, and `ollama create shoptalk-llama31 -f Modelfile`) —
at that point it's a drop-in replacement: set `llm.model:
shoptalk-llama31` in `configs/config.yaml` and every other component
(`rag/chain.py`, the API, the UI) picks it up unchanged.

## Evaluating the fine-tune

Compare base vs. fine-tuned on the golden set using the same LLM-as-judge
pattern the design doc recommends for RAG quality (§6, §12.4): sample N
golden-set queries, generate with both models, score with a stronger judge
model (or human review) on faithfulness/relevance/helpfulness. Not executed
here — requires an actual trained adapter to compare against.
