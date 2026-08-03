"""
Stage-2 cross-encoder reranker. Scores (query, candidate_document) pairs
jointly -- far more accurate than the stage-1 bi-encoder because query and
document attend to each other, but too slow to run over the full catalog, so
it only ever sees the stage-1 top-K candidates.
"""
from sentence_transformers import CrossEncoder

from shoptalk.config import load_config

_cross_encoder = None


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def rerank(query: str, candidates: list, top_k: int = None, cfg: dict = None) -> list:
    """
    candidates: list of hit dicts as produced by retrieval.search / two_stage
                (must have a "document" key). Returns the same dicts, sorted
                by rerank_score descending, truncated to top_k, with a
                "rerank_score" field added and "stage1_score" preserving the
                original bi-encoder similarity for comparison.
    """
    cfg = cfg or load_config()
    rcfg = cfg["retrieval"]
    top_k = top_k or cfg["retrieval"]["top_k"]

    if not candidates:
        return []

    model = _get_cross_encoder(rcfg["rerank_model"])
    pairs = [(query, c["document"]) for c in candidates]
    scores = model.predict(pairs, batch_size=rcfg["rerank_batch_size"], show_progress_bar=False)

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["stage1_score"] = item.get("score", item.get("stage1_score"))
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
