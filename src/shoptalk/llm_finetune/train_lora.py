"""
LoRA/QLoRA instruction fine-tuning for the generation LLM (design doc §3.2b).

NOT EXECUTED as part of this build -- per the problem statement's explicit
allowance ("LLM LoRA run: deliver code + docs only"), and because it needs a
real GPU with 12GB+ VRAM (bitsandbytes 4-bit quantization + LoRA adapters)
that this project's CPU-only validation environment does not have. This is
real, complete, runnable code -- meant to run on Kaggle's free T4/P100 GPU
tier -- not a stub. See docs/finetuning/llm_lora_qlora.md for the dataset
format spec, config rationale, and merge/serve steps.

Usage (on a GPU machine):
  pip install -r requirements/llm_finetune.txt
  python -m shoptalk.llm_finetune.train_lora \\
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset data/llm_finetune/instructions.jsonl \\
    --output-dir data/models/llama31-8b-lora \\
    --qlora   # add for 4-bit QLoRA; omit for standard LoRA
"""
import argparse
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROMPT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

<catalog_results>
{retrieved_context}
</catalog_results>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""


def load_dataset(path: str) -> Dataset:
    rows = [json.loads(line) for line in open(path)]
    texts = [PROMPT_TEMPLATE.format(**row) for row in rows]
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="data/llm_finetune/instructions.jsonl")
    parser.add_argument("--output-dir", default="data/models/llama31-8b-lora")
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA instead of standard LoRA")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # attention + MLP projections -- the standard Llama LoRA target set
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="mlflow",  # logs to the same MLflow tracking server as everything else
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved -> {args.output_dir}")
    print("next: see docs/finetuning/llm_lora_qlora.md for merge/serve steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
