"""
Fine-tunes the text bi-encoder (bge-base-en-v1.5) with triplet loss and
hard-negative mining (design doc §3.1, §12.3): anchor = golden-set query,
positive = its labeled relevant product's document, negative = a hard
negative mined from the SAME category via the current (pre-finetune) index
-- visually/textually similar but wrong, which is what actually teaches the
model fine-grained distinctions (random negatives from unrelated categories
are already easy to separate and add little signal).

Reuses the Day-2 golden eval set as training supervision rather than
requiring separate labels -- the same query/positive-product pairs already
curated for evaluation double as fine-tuning triplets.

Resumable: trains ONE epoch at a time, saving the model to --output-dir and
recording progress in <output-dir>/_finetune_state.json after every epoch
(same Colab-disconnect motivation as the checkpointed captioning/embedding
steps -- a disconnect in epoch 3 of 3 used to lose all training). Re-running
the same command loads the last saved epoch's weights and continues with the
remaining epochs. The state file is removed on clean completion.

Usage:
  python -m shoptalk.embeddings.finetune_text
  python -m shoptalk.embeddings.finetune_text --epochs 1   # smoke test
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from shoptalk.config import load_config

# Set here (module level, after imports) rather than before the imports
# above -- these are read lazily by wandb/tokenizers/torch when actually
# used (model construction, .fit()), not by transformers'/sentence_
# transformers' own imports, so there's no need to set them before those
# imports run (which would also trip ruff's import-order check). They just
# need to be set before main() constructs a model or calls .fit(), which
# this satisfies.
#
# WANDB_DISABLED/WANDB_MODE: sentence_transformers' .fit() wraps HuggingFace's
# Trainer, which auto-detects installed integrations and defaults to
# reporting to all of them. wandb is installed (via requirements/eval.txt)
# but never logged in, so the first .fit() call means an interactive
# "(1) Create a W&B account (2) Use an existing account (3) Don't visualize"
# prompt -- caught for real on a Colab run: it hung the cell waiting for
# input nobody was there to give, indistinguishable from a stalled/frozen
# step from the outside. This is an unattended batch pipeline; it must never
# block on stdin.
#
# PYTORCH_CUDA_ALLOC_CONF: directly suggested by a real CUDA OOM caught on a
# T4 (14.55/14.56 GiB in use, failed to allocate 34 more MiB) -- reduces
# allocator fragmentation, which can be the difference between an allocation
# succeeding or not when memory is this tight.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def mine_hard_negative(pos_id: str, category: str, exclude_ids: set, products_df: pd.DataFrame, rng: random.Random):
    same_cat = products_df[(products_df["category"] == category) & (~products_df["item_id"].isin(exclude_ids))]
    if same_cat.empty:
        same_cat = products_df[~products_df["item_id"].isin(exclude_ids)]
    if same_cat.empty:
        return None
    return same_cat.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]


def build_triplets(golden_set: list, products_df: pd.DataFrame, seed: int) -> list:
    rng = random.Random(seed)
    doc_by_id = products_df.set_index("item_id")["document"]
    cat_by_id = products_df.set_index("item_id")["category"]

    examples = []
    for row in golden_set:
        if not row["positive_ids"]:
            continue
        pos_id = row["positive_ids"][0]
        if pos_id not in doc_by_id.index:
            continue
        negative = mine_hard_negative(pos_id, cat_by_id[pos_id], set(row["positive_ids"]), products_df, rng)
        if negative is None:
            continue
        examples.append(InputExample(texts=[row["query"], doc_by_id[pos_id], negative["document"]]))
    return examples


def _load_state(state_path: Path) -> int:
    """Completed-epoch count from a previous interrupted run (0 if none).
    Only trusts the state file if the saved model directory is actually
    loadable (config present) -- guards against a half-written save."""
    if not state_path.exists():
        return 0
    try:
        state = json.loads(state_path.read_text())
        completed = int(state.get("completed_epochs", 0))
    except (ValueError, json.JSONDecodeError):
        return 0
    model_config = state_path.parent / "config.json"
    if completed > 0 and not model_config.exists():
        return 0
    return completed


def _save_state(state_path: Path, completed_epochs: int, total_epochs: int) -> None:
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"completed_epochs": completed_epochs, "total_epochs": total_epochs}))
    os.replace(tmp, state_path)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="overrides embeddings.device from config.yaml -- e.g. --device cpu to avoid "
        "GPU contention when something else (Ollama, a concurrent embedding job) is already "
        "holding the GPU. This step is small (~100 triplets) so CPU is plenty fast here.",
    )
    args = parser.parse_args()

    cfg = load_config()
    dcfg, ecfg = cfg["data"], cfg["embeddings"]
    device = resolve_device(args.device or ecfg.get("device", "auto"))
    print(f"device: {device}")

    golden_path = Path(cfg["eval"]["golden_set_path"])
    if not golden_path.exists():
        print(f"error: {golden_path} not found -- run generate_golden_set.py first", file=sys.stderr)
        return 1

    golden_set = [json.loads(line) for line in open(golden_path)]
    products_df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")

    examples = build_triplets(golden_set, products_df, dcfg["seed"])
    print(f"built {len(examples)} triplets from {len(golden_set)} golden queries")
    if len(examples) < 10:
        print(
            "warning: very few triplets -- fine-tune quality will be weak at this "
            "scale (expected for a smoke test; the full ~10k catalog + 100-query "
            "golden set gives a much richer triplet set)"
        )

    output_dir = args.output_dir or str(Path(dcfg["processed_dir"]).parent / "models" / "bge-finetuned")
    state_path = Path(output_dir) / "_finetune_state.json"
    completed = _load_state(state_path)

    if completed >= args.epochs:
        print(f"already trained {completed}/{args.epochs} epochs (per {state_path}) -- nothing left to run")
        state_path.unlink(missing_ok=True)
        return 0

    if completed > 0:
        print(f"resuming from epoch {completed + 1}/{args.epochs} (loading saved weights from {output_dir})")
        model = SentenceTransformer(output_dir, device=device)
    else:
        # Always the pristine pretrained model, NOT ecfg["text_model"] -- the
        # latter may itself already point at a fine-tuned checkpoint (once one
        # has been promoted to serving), and fine-tuning from itself on the
        # same golden set again is not a meaningful fresh retrain.
        base_text_model = ecfg.get("base_text_model", ecfg["text_model"])
        model = SentenceTransformer(base_text_model, device=device)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(model=model)

    for epoch in range(completed, args.epochs):
        # Between epochs in the same process -- cheap and harmless if there's
        # nothing to reclaim, but real memory was this tight on a T4 (see the
        # PYTORCH_CUDA_ALLOC_CONF note above): 14.55/14.56 GiB in use.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # warmup only at the very start of a fresh run -- a resumed model is
        # already past the unstable early-LR phase
        warmup = max(1, int(0.1 * len(train_dataloader))) if epoch == 0 else 0
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup,
            output_path=output_dir,  # saves model at end of this epoch
            show_progress_bar=True,
        )
        _save_state(state_path, epoch + 1, args.epochs)
        print(f"epoch {epoch + 1}/{args.epochs} complete -- checkpoint saved to {output_dir}", flush=True)

    state_path.unlink(missing_ok=True)  # clean completion
    print(f"fine-tuned model saved -> {output_dir}")
    print("next: python -m shoptalk.embeddings.eval_finetune --finetuned-path " + output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
