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

Usage:
  python -m shoptalk.data.caption_images
  python -m shoptalk.data.caption_images --limit 200   # smoke test
"""
import argparse
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap rows captioned (smoke tests)")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, ccfg = cfg["data"], cfg["captioning"]

    df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")
    if args.limit:
        df = df.head(args.limit).copy()

    device = resolve_device(ccfg["device"])
    print(f"captioning {len(df)} products with {ccfg['model']} on {device}")

    processor = BlipProcessor.from_pretrained(ccfg["model"])
    model = BlipForConditionalGeneration.from_pretrained(ccfg["model"]).to(device)
    model.eval()

    captions = [""] * len(df)
    batch_size = ccfg["batch_size"]
    rows = df.reset_index(drop=True)

    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        paths = [p if avail else None for p, avail in zip(batch["image_path"], batch["image_available"])]
        # skip PIL.Image.open on None paths -- caption_batch already handles
        # FileNotFoundError, but None isn't a valid path type, so pre-filter
        batch_captions = [""] * len(batch)
        real_paths = [(i, p) for i, p in enumerate(paths) if p]
        if real_paths:
            idxs, real_path_list = zip(*real_paths)
            sub_captions = caption_batch(processor, model, device, list(real_path_list), ccfg["max_new_tokens"])
            for local_i, caption in zip(idxs, sub_captions):
                batch_captions[local_i] = caption

        captions[start : start + len(batch)] = batch_captions
        # flush=True matters here: under `!python -m ...` in a notebook cell,
        # stdout isn't a TTY, so Python fully block-buffers it -- without an
        # explicit flush, this line sits invisible for a long stretch (until
        # the OS pipe buffer fills or the process exits) rather than updating
        # live, making a genuinely-progressing run look frozen.
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)} captioned", end="\r", flush=True)

    print()
    df["caption"] = captions
    has_caption = df["caption"] != ""
    df.loc[has_caption, "document"] = df.loc[has_caption, "document"] + " | " + df.loc[has_caption, "caption"]

    processed_dir = dcfg["processed_dir"]
    df.to_parquet(f"{processed_dir}/products.parquet", index=False)
    df.to_csv(f"{processed_dir}/products.csv", index=False)

    print(f"captioned {has_caption.sum()}/{len(df)} products (rest had no available image)")
    print(f"example: {df.loc[has_caption, 'caption'].iloc[0]!r}" if has_caption.any() else "no captions generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
