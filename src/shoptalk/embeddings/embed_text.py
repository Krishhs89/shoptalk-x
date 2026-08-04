"""
Embeds the `document` field of every processed product with bge-base-en-v1.5
and upserts the vectors into a persistent Chroma collection.

BGE convention: passages (our product documents) are embedded as-is; queries
get an instruction prefix at *search* time (see retrieval/search.py), not here.

Resumable: each completed batch's embeddings are appended to a
data/processed/_text_embeddings_checkpoint.jsonl file and fsync'd to disk
before moving to the next batch -- same pattern as embed_image.py /
caption_images.py, same motivation (a Colab runtime disconnect partway
through a 10k-product run used to lose everything, because the old code did
one monolithic model.encode() and only wrote to Chroma at the very end).
Re-running the exact same command skips every item_id already in the
checkpoint. The Chroma collection is only (re)built once, from the full
merged set, at the end -- the checkpoint is deleted on a clean completion.

Input:  data/processed/products.parquet
Output: data/chroma/  (persistent Chroma collection, see configs/config.yaml)

Usage:
  python -m shoptalk.embeddings.embed_text
  python -m shoptalk.embeddings.embed_text --limit 200   # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from shoptalk.config import load_config


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_checkpoint(checkpoint_path: Path) -> dict:
    """item_id -> embedding for every batch already completed. Last write
    wins on duplicate ids (can only happen if a batch was re-run)."""
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
    dcfg, ecfg = cfg["data"], cfg["embeddings"]

    df = pd.read_parquet(f"{dcfg['processed_dir']}/products.parquet")
    if args.limit:
        df = df.head(args.limit)
    print(f"embedding {len(df)} products with {ecfg['text_model']}")

    checkpoint_path = Path(dcfg["processed_dir"]) / "_text_embeddings_checkpoint.jsonl"
    embedded = _load_checkpoint(checkpoint_path)
    # Guard against a checkpoint from a different --limit/scope: only reuse
    # entries that are still part of this run's product set.
    embedded = {k: v for k, v in embedded.items() if k in set(df["item_id"])}
    if embedded:
        print(f"resuming from checkpoint: {len(embedded)}/{len(df)} products already embedded")

    pending_df = df[~df["item_id"].isin(embedded.keys())].reset_index(drop=True)

    if pending_df.empty:
        print("all products already embedded in checkpoint -- nothing left to run")
    else:
        device = resolve_device(ecfg["device"])
        print(f"device: {device}")
        model = SentenceTransformer(ecfg["text_model"], device=device)

        batch_size = ecfg["batch_size"]
        with open(checkpoint_path, "a") as ckpt_f:
            for start in range(0, len(pending_df), batch_size):
                batch = pending_df.iloc[start : start + batch_size]
                vectors = model.encode(
                    batch["document"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # cosine similarity via dot product
                )
                for item_id, vec in zip(batch["item_id"], vectors):
                    vec_list = vec.tolist()
                    ckpt_f.write(json.dumps({"item_id": item_id, "embedding": vec_list}) + "\n")
                    embedded[item_id] = vec_list
                ckpt_f.flush()
                import os

                os.fsync(ckpt_f.fileno())
                print(
                    f"  embedded {min(start + batch_size, len(pending_df))}/{len(pending_df)} pending "
                    f"({len(embedded)}/{len(df)} total)",
                    flush=True,
                )

    client = chromadb.PersistentClient(path=ecfg["chroma_dir"])
    # fresh collection each run so re-embedding never mixes stale + new vectors
    try:
        client.delete_collection(ecfg["collection_name"])
    except Exception:
        pass
    collection = client.create_collection(
        name=ecfg["collection_name"], metadata={"hnsw:space": "cosine"}
    )

    metadatas = df[["item_id", "item_name", "category", "brand", "price_usd", "image_path"]].copy()
    metadatas["image_path"] = metadatas["image_path"].fillna("")
    metadatas["price_usd"] = metadatas["price_usd"].astype(float)

    collection.add(
        ids=df["item_id"].tolist(),
        embeddings=[embedded[item_id] for item_id in df["item_id"]],
        documents=df["document"].tolist(),
        metadatas=metadatas.to_dict("records"),
    )

    print(f"indexed {collection.count()} vectors -> {ecfg['chroma_dir']}/{ecfg['collection_name']}")
    checkpoint_path.unlink(missing_ok=True)  # clean completion -- avoid stale-resume confusion
    return 0


if __name__ == "__main__":
    sys.exit(main())
