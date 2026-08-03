"""
Day-2 two-stage retrieval: stage-1 ANN (bi-encoder, top-100) -> stage-2
cross-encoder rerank (top-100 -> top-K). See
docs/ShopTalk-X_Production_Design_Document.md §3.2.

Usage:
  python -m shoptalk.retrieval.two_stage --query "red shirt for men under 50 dollars"
"""
import argparse
import sys

from shoptalk.config import load_config
from shoptalk.retrieval.rerank import rerank
from shoptalk.retrieval.search import search as stage1_search


def two_stage_search(query: str, stage1_k: int = None, top_k: int = None, cfg: dict = None) -> list:
    cfg = cfg or load_config()
    rcfg = cfg["retrieval"]
    stage1_k = stage1_k or rcfg["stage1_k"]
    top_k = top_k or rcfg["top_k"]

    candidates = stage1_search(query, top_k=stage1_k, cfg=cfg)
    return rerank(query, candidates, top_k=top_k, cfg=cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stage1-k", type=int, default=None)
    args = parser.parse_args()

    hits = two_stage_search(args.query, stage1_k=args.stage1_k, top_k=args.top_k)
    print(f"query: {args.query!r} -> {len(hits)} results (two-stage)\n")
    for rank, hit in enumerate(hits, 1):
        md = hit["metadata"]
        print(
            f"{rank:2d}. [rerank={hit['rerank_score']:.3f} stage1={hit['stage1_score']:.3f}] "
            f"{md['item_name'][:65]!r} ({md['category']}, ${md['price_usd']:.2f}) -- id={hit['item_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
