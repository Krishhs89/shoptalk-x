"""
Post-retrieval personalization rerank blend (design doc §3.2c):
final_score = alpha * cross-encoder_score + beta * cosine(user_profile, item_embedding)

Applied AFTER the existing two-stage retrieval + rerank (retrieval.rerank) --
this never replaces the base ranking, only nudges it toward a user's
demonstrated taste. A user with no interaction history (or none of their
interacted items embedded) falls back to the unpersonalized ranking
unchanged.

Usage:
  python -m shoptalk.personalization.rerank_blend --query "shoes" --user-id user_0001
"""
import argparse
import sys

import chromadb
import numpy as np
import pandas as pd

from shoptalk.config import load_config
from shoptalk.personalization.profile import build_user_profile
from shoptalk.retrieval.two_stage import two_stage_search


def _min_max_normalize(scores: list) -> list:
    arr = np.array(scores, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()


def personalized_rerank(query: str, user_id: str, top_k: int = None, cfg: dict = None) -> list:
    cfg = cfg or load_config()
    pcfg, ecfg = cfg["personalization"], cfg["embeddings"]
    top_k = top_k or cfg["retrieval"]["top_k"]

    hits = two_stage_search(query, top_k=cfg["retrieval"]["stage1_k"], cfg=cfg)
    if not hits:
        return []

    client = chromadb.PersistentClient(path=ecfg["chroma_dir"])
    collection = client.get_collection(ecfg["collection_name"])

    hit_ids = [h["item_id"] for h in hits]
    result = collection.get(ids=hit_ids, include=["embeddings"])
    hit_embeddings = {iid: np.array(emb) for iid, emb in zip(result["ids"], result["embeddings"])}

    interactions_df = pd.read_json(pcfg["interactions_path"], lines=True)
    # dedupe -- a user can click/purchase the same item more than once, and
    # Chroma's get() rejects a request containing duplicate ids
    user_interacted_ids = interactions_df[interactions_df["user_id"] == user_id]["item_id"].unique().tolist()
    profile_embeddings = (
        collection.get(ids=user_interacted_ids, include=["embeddings"]) if user_interacted_ids else None
    )
    item_embeddings = (
        {iid: np.array(emb) for iid, emb in zip(profile_embeddings["ids"], profile_embeddings["embeddings"])}
        if profile_embeddings
        else {}
    )
    profile = build_user_profile(user_id, interactions_df, item_embeddings, pcfg["decay"])

    if profile is None:
        # no usable history -- unpersonalized ranking, unchanged
        return hits[:top_k]

    rerank_scores = _min_max_normalize([h["rerank_score"] for h in hits])
    blended = []
    for hit, norm_rerank in zip(hits, rerank_scores):
        item_emb = hit_embeddings.get(hit["item_id"])
        profile_sim = float(np.dot(profile, item_emb)) if item_emb is not None else 0.0
        final_score = pcfg["alpha"] * norm_rerank + pcfg["beta"] * profile_sim
        item = dict(hit)
        item["profile_similarity"] = profile_sim
        item["personalized_score"] = final_score
        blended.append(item)

    blended.sort(key=lambda x: x["personalized_score"], reverse=True)
    return blended[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    hits = personalized_rerank(args.query, args.user_id, top_k=args.top_k)
    print(f"query: {args.query!r}  user: {args.user_id}\n")
    for rank, hit in enumerate(hits, 1):
        md = hit["metadata"]
        extra = (
            f"personalized={hit['personalized_score']:.3f} profile_sim={hit['profile_similarity']:.3f}"
            if "personalized_score" in hit
            else "(unpersonalized -- no usable history)"
        )
        print(f"{rank:2d}. [{extra}] {md['item_name'][:60]!r} ({md['category']}) -- id={hit['item_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
