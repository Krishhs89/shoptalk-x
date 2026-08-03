"""
Embeds the `document` field of every processed product with bge-base-en-v1.5
and upserts the vectors into a persistent Chroma collection.

BGE convention: passages (our product documents) are embedded as-is; queries
get an instruction prefix at *search* time (see retrieval/search.py), not here.

Input:  data/processed/products.parquet
Output: data/chroma/  (persistent Chroma collection, see configs/config.yaml)

Usage:
  python -m shoptalk.embeddings.embed_text
  python -m shoptalk.embeddings.embed_text --limit 200   # smoke test
"""
import argparse
import sys

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

    device = resolve_device(ecfg["device"])
    print(f"device: {device}")
    model = SentenceTransformer(ecfg["text_model"], device=device)

    embeddings = model.encode(
        df["document"].tolist(),
        batch_size=ecfg["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
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
        embeddings=embeddings.tolist(),
        documents=df["document"].tolist(),
        metadatas=metadatas.to_dict("records"),
    )

    print(f"indexed {collection.count()} vectors -> {ecfg['chroma_dir']}/{ecfg['collection_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
