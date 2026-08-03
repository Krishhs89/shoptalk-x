"""
Day-1 baseline retrieval: text query -> top-K ANN results from the Chroma
index built by embeddings/embed_text.py. Single-stage (no reranker yet --
that's Day 2).

Usage:
  python -m shoptalk.retrieval.search --query "red shirt for men"
  python -m shoptalk.retrieval.search --query "wireless headphones" --top-k 5
"""
import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from shoptalk.config import load_config

# BGE 1.5's recommended asymmetric-retrieval instruction: prepended to QUERIES
# only, never to the indexed passages (see embed_text.py).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_collection = None


def _get_model(model_name: str):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def _get_collection(chroma_dir: str, collection_name: str):
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=chroma_dir)
        _collection = client.get_collection(collection_name)
    return _collection


def search(query: str, top_k: int = None, cfg: dict = None) -> list:
    cfg = cfg or load_config()
    ecfg, rcfg = cfg["embeddings"], cfg["retrieval"]
    top_k = top_k or rcfg["top_k"]

    model = _get_model(ecfg["text_model"])
    collection = _get_collection(ecfg["chroma_dir"], ecfg["collection_name"])

    query_embedding = model.encode(
        [BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "item_id": results["ids"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance -> similarity
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
            }
        )
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    hits = search(args.query, top_k=args.top_k)
    print(f"query: {args.query!r} -> {len(hits)} results\n")
    for rank, hit in enumerate(hits, 1):
        md = hit["metadata"]
        print(
            f"{rank:2d}. [{hit['score']:.3f}] {md['item_name'][:70]!r} "
            f"({md['category']}, {md['brand']}, ${md['price_usd']:.2f}) -- id={hit['item_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
