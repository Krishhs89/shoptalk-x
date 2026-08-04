"""
Embeds every catalog product image with OpenCLIP (LAION) and upserts into a
second, dedicated Chroma collection (image embedding space != text embedding
space -- different dims, not directly comparable -- see design doc §3.1).

Input:  data/processed/products.parquet (image_path, image_available columns)
Output: data/chroma/ collection `clip.collection_name` (default: shoptalk_images)

Resumable: each completed batch's embeddings are appended to a
data/processed/_image_embeddings_checkpoint.jsonl file and fsync'd to disk
before moving to the next batch. If the process dies partway through (e.g.
a Colab runtime disconnect), re-running this exact command skips every
item_id already in the checkpoint instead of re-running CLIP on it. The
Chroma collection itself is only (re)built once, from the full merged set,
at the end -- the checkpoint is deleted on a clean full completion.

Usage:
  python -m shoptalk.embeddings.embed_image
  python -m shoptalk.embeddings.embed_image --limit 200   # smoke test
"""
import argparse
import json
import os
import sys
from pathlib import Path

import chromadb
import open_clip
import pandas as pd
import torch
from PIL import Image

from shoptalk.config import load_config


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_checkpoint(checkpoint_path: Path) -> dict:
    embedded = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                embedded[rec["item_id"]] = rec["embedding"]
    return embedded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap rows embedded (smoke tests)")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, clcfg = cfg["data"], cfg["clip"]

    df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")
    df = df[df["image_available"]].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"embedding {len(df)} product images with OpenCLIP {clcfg['model_name']} ({clcfg['pretrained']})")

    checkpoint_path = Path(dcfg["processed_dir"]) / "_image_embeddings_checkpoint.jsonl"
    embedded = _load_checkpoint(checkpoint_path)
    if embedded:
        print(f"resuming from checkpoint: {len(embedded)}/{len(df)} images already embedded")

    pending_df = df[~df["item_id"].isin(embedded.keys())].reset_index(drop=True)

    if pending_df.empty:
        print("all images already embedded in checkpoint -- nothing left to run")
    else:
        device = resolve_device(clcfg["device"])
        print(f"device: {device}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            clcfg["model_name"], pretrained=clcfg["pretrained"], device=device
        )
        model.eval()

        batch_size = clcfg["batch_size"]
        with open(checkpoint_path, "a") as ckpt_f:
            for start in range(0, len(pending_df), batch_size):
                batch = pending_df.iloc[start : start + batch_size]
                images, kept_item_ids = [], []
                for item_id, path in zip(batch["item_id"], batch["image_path"]):
                    try:
                        images.append(preprocess(Image.open(path).convert("RGB")))
                        kept_item_ids.append(item_id)
                    except (FileNotFoundError, OSError):
                        continue
                if not images:
                    continue

                pixel_values = torch.stack(images).to(device)
                with torch.no_grad():
                    batch_embeddings = model.encode_image(pixel_values)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)  # cosine via dot

                for item_id, vec in zip(kept_item_ids, batch_embeddings.cpu().numpy()):
                    vec_list = vec.tolist()
                    embedded[item_id] = vec_list
                    ckpt_f.write(json.dumps({"item_id": item_id, "embedding": vec_list}) + "\n")
                # flush + fsync so a completed batch survives a hard interruption
                # (e.g. a Colab runtime disconnect), not just a clean process exit
                # -- flush=True on the progress print matters too: under
                # `!python -m ...` in a notebook cell, stdout isn't a TTY, so an
                # unflushed line can sit invisible for a long stretch, making a
                # genuinely-progressing run look frozen.
                ckpt_f.flush()
                os.fsync(ckpt_f.fileno())

                done = min(start + batch_size, len(pending_df))
                print(f"  {done}/{len(pending_df)} embedded", end="\r", flush=True)
        print()

    embedded_df = df[df["item_id"].isin(embedded.keys())].reset_index(drop=True)
    embeddings_list = [embedded[item_id] for item_id in embedded_df["item_id"]]

    client = chromadb.PersistentClient(path=clcfg["chroma_dir"])
    try:
        client.delete_collection(clcfg["collection_name"])
    except Exception:
        pass
    collection = client.create_collection(name=clcfg["collection_name"], metadata={"hnsw:space": "cosine"})

    metadatas = embedded_df[["item_id", "item_name", "category", "brand", "price_usd", "image_path"]].copy()
    metadatas["price_usd"] = metadatas["price_usd"].astype(float)

    collection.add(
        ids=embedded_df["item_id"].tolist(),
        embeddings=embeddings_list,
        documents=embedded_df["document"].tolist(),
        metadatas=metadatas.to_dict("records"),
    )

    print(f"indexed {collection.count()} image vectors -> {clcfg['chroma_dir']}/{clcfg['collection_name']}")
    checkpoint_path.unlink(missing_ok=True)  # clean completion -- avoid stale-resume confusion on a future fresh run
    return 0


if __name__ == "__main__":
    sys.exit(main())
