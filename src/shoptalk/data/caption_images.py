"""
Day-3 captioning batch job: runs BLIP over every product's main image and
appends the caption to both a dedicated `caption` column and the `document`
field products get embedded from -- captures visual attributes (pattern,
shape, material appearance) that sellers often omit from bullet points (see
design doc §2, "Augmented text").

Products with no available image get caption="" and an unchanged document.

Input:  data/processed/products.parquet (from preprocess.py)
Output: data/processed/products.parquet + .csv, overwritten in place with the
        new `caption` column and caption-augmented `document` field.

IMPORTANT: run this BEFORE re-running embeddings.embed_text, so the index
reflects the caption-augmented documents.

Resumable: each completed batch is appended to a
data/processed/_captions_checkpoint.jsonl file and fsync'd to disk before
moving to the next batch. If the process dies partway through (e.g. a Colab
runtime disconnect on a large run), re-running this exact command skips
every item_id already in the checkpoint instead of re-captioning from
scratch. The checkpoint is deleted on a clean full completion.

Usage:
  python -m shoptalk.data.caption_images
  python -m shoptalk.data.caption_images --limit 200   # smoke test
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from shoptalk.config import load_config


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def caption_batch(processor, model, device, image_paths: list, max_new_tokens: int) -> list:
    images, valid_idx = [], []
    for i, path in enumerate(image_paths):
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_idx.append(i)
        except (FileNotFoundError, OSError):
            continue

    captions = [""] * len(image_paths)
    if not images:
        return captions

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        # repetition_penalty + no_repeat_ngram_size guard against BLIP's greedy-decoding
        # failure mode of looping on a single token ("sailor sailor sailor ...")
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
    decoded = processor.batch_decode(out, skip_special_tokens=True)

    for idx, caption in zip(valid_idx, decoded):
        captions[idx] = caption.strip()
    return captions


def _load_checkpoint(checkpoint_path: Path) -> dict:
    captioned = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                captioned[rec["item_id"]] = rec["caption"]
    return captioned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap rows captioned (smoke tests)")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, ccfg = cfg["data"], cfg["captioning"]
    processed_dir = dcfg["processed_dir"]

    df = pd.read_parquet(f"{processed_dir}/products.parquet")
    if args.limit:
        df = df.head(args.limit).copy()
    rows = df.reset_index(drop=True)

    checkpoint_path = Path(processed_dir) / "_captions_checkpoint.jsonl"
    captioned = _load_checkpoint(checkpoint_path)
    if captioned:
        print(f"resuming from checkpoint: {len(captioned)}/{len(rows)} products already captioned")

    pending_mask = ~rows["item_id"].isin(captioned.keys())
    pending_rows = rows[pending_mask]

    if pending_rows.empty:
        print("all products already captioned in checkpoint -- nothing left to run")
    else:
        device = resolve_device(ccfg["device"])
        print(f"captioning {len(pending_rows)} remaining products with {ccfg['model']} on {device}")

        processor = BlipProcessor.from_pretrained(ccfg["model"])
        model = BlipForConditionalGeneration.from_pretrained(ccfg["model"]).to(device)
        model.eval()

        batch_size = ccfg["batch_size"]

        with open(checkpoint_path, "a") as ckpt_f:
            for start in range(0, len(pending_rows), batch_size):
                batch = pending_rows.iloc[start : start + batch_size]
                paths = [p if avail else None for p, avail in zip(batch["image_path"], batch["image_available"])]
                # skip PIL.Image.open on None paths -- caption_batch already handles
                # FileNotFoundError, but None isn't a valid path type, so pre-filter
                batch_captions = [""] * len(batch)
                real_paths = [(i, p) for i, p in enumerate(paths) if p]
                if real_paths:
                    idxs, real_path_list = zip(*real_paths)
                    sub_captions = caption_batch(
                        processor, model, device, list(real_path_list), ccfg["max_new_tokens"]
                    )
                    for local_i, caption in zip(idxs, sub_captions):
                        batch_captions[local_i] = caption

                for item_id, caption in zip(batch["item_id"], batch_captions):
                    captioned[item_id] = caption
                    ckpt_f.write(json.dumps({"item_id": item_id, "caption": caption}) + "\n")
                # flush + fsync so a completed batch survives a hard interruption
                # (e.g. a Colab runtime disconnect), not just a clean process exit
                # -- flush=True on the progress print matters too: under
                # `!python -m ...` in a notebook cell, stdout isn't a TTY, so
                # Python fully block-buffers it and an unflushed line can sit
                # invisible for a long stretch, making a genuinely-progressing
                # run look frozen.
                ckpt_f.flush()
                os.fsync(ckpt_f.fileno())

                done = min(start + batch_size, len(pending_rows))
                print(f"  {done}/{len(pending_rows)} captioned", end="\r", flush=True)
        print()

    df = rows
    df["caption"] = [captioned.get(item_id, "") for item_id in df["item_id"]]
    has_caption = df["caption"] != ""
    df.loc[has_caption, "document"] = df.loc[has_caption, "document"] + " | " + df.loc[has_caption, "caption"]

    df.to_parquet(f"{processed_dir}/products.parquet", index=False)
    df.to_csv(f"{processed_dir}/products.csv", index=False)
    checkpoint_path.unlink(missing_ok=True)  # clean completion -- avoid stale-resume confusion on a future fresh run

    print(f"captioned {has_caption.sum()}/{len(df)} products (rest had no available image)")
    print(f"example: {df.loc[has_caption, 'caption'].iloc[0]!r}" if has_caption.any() else "no captions generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
