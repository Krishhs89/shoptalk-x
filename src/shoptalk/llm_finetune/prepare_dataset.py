"""
Generates a synthetic instruction-tuning dataset for the generation LLM
(design doc §3.2b: "dataset format spec (instruction/query/retrieved-
context/response)"). Reuses the actual production retrieval pipeline (Day 2
two-stage search) so the training data's context distribution matches what
the model sees at serving time, then has the (already-deployed) LLM itself
write the target response -- teacher-generated instruction data is standard
practice when no human-written response corpus exists for a niche domain.

The resulting JSONL is the input to train_lora.py. Executing the LoRA/QLoRA
training run itself is out of scope for this build (per the problem
statement's explicit allowance: "LLM LoRA run: deliver code + docs only") --
this script IS run for real, producing genuine (if small-scale) training
examples; only the GPU training step is documented-not-executed.

Usage:
  python -m shoptalk.llm_finetune.prepare_dataset --n 20
"""
import argparse
import json
import sys
from pathlib import Path

from shoptalk.config import load_config
from shoptalk.rag.chain import answer_from_hits
from shoptalk.rag.prompts import format_catalog_block
from shoptalk.retrieval.two_stage import two_stage_search

INSTRUCTION = (
    "You are ShopTalk, a shopping assistant. Given the user's question and the "
    "retrieved catalog products below, answer helpfully and cite product IDs."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="number of examples to generate")
    parser.add_argument(
        "--output", default=None, help="output path (default: data/llm_finetune/instructions.jsonl)"
    )
    args = parser.parse_args()

    cfg = load_config()
    golden_path = Path(cfg["eval"]["golden_set_path"])
    if not golden_path.exists():
        print(f"error: {golden_path} not found -- run generate_golden_set.py first", file=sys.stderr)
        return 1

    golden_set = [json.loads(line) for line in open(golden_path)][: args.n]
    default_output = Path(cfg["data"]["raw_dir"]).parent / "llm_finetune" / "instructions.jsonl"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for i, row in enumerate(golden_set):
        query = row["query"]
        hits = two_stage_search(query, cfg=cfg)
        context = format_catalog_block(hits[: cfg["llm"]["context_products"]])
        response = answer_from_hits(query, hits, history=[], cfg=cfg, stream=False)

        examples.append(
            {
                "instruction": INSTRUCTION,
                "query": query,
                "retrieved_context": context,
                "response": response,
            }
        )
        print(f"[{i + 1}/{len(golden_set)}] {query!r} -> {len(response)} char response")

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nwrote {len(examples)} instruction examples -> {output_path}")
    print("next: see docs/finetuning/llm_lora_qlora.md for the (Kaggle-GPU) training step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
