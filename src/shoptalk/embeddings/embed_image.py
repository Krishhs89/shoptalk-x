"""
Embeds every catalog product image with OpenCLIP (LAION) and upserts into a
second, dedicated Chroma collection (image embedding space != text embedding
space -- different dims, not directly comparable -- see design doc §3.1).

Input:  data/processed/products.parquet (image_path, image_available columns)
Output: data/chroma/ collection `clip.collection_name` (default: shoptalk_images)

Usage:
  python -m shoptalk.embeddings.embed_image
  python -m shoptalk.embeddings.embed_image --limit 200   # smoke test
"""
import argparse
import sys

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

    device = resolve_device(clcfg["device"])
    print(f"device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clcfg["model_name"], pretrained=clcfg["pretrained"], device=device
    )
    model.eval()

    all_embeddings, valid_rows = [], []
    batch_size = clcfg["batch_size"]
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        images, kept = [], []
        for i, path in zip(batch.index, batch["image_path"]):
            try:
                images.append(preprocess(Image.open(path).convert("RGB")))
                kept.append(i)
            except (FileNotFoundError, OSError):
                continue
        if not images:
            continue

        pixel_values = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(pixel_values)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # cosine via dot product

        all_embeddings.append(embeddings.cpu())
        valid_rows.extend(kept)
        print(f"  {min(start + batch_size, len(df))}/{len(df)} embedded", end="\r")

    print()
    embeddings = torch.cat(all_embeddings).numpy()
    embedded_df = df.loc[valid_rows].reset_index(drop=True)

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
        embeddings=embeddings.tolist(),
        documents=embedded_df["document"].tolist(),
        metadatas=metadatas.to_dict("records"),
    )

    print(f"indexed {collection.count()} image vectors -> {clcfg['chroma_dir']}/{clcfg['collection_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
