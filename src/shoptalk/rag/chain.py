"""
Day-4 RAG chain: two-stage retriever -> prompt template -> self-hosted LLM
(Ollama), with manual conversation-history injection for follow-up support.

Conversation memory is implemented as an explicit list of prior turns passed
in by the caller (API layer owns the session state) rather than a LangChain
Memory object -- those classes are legacy/soft-deprecated in current
LangChain, and explicit state is easier to reason about and to serve
statelessly from FastAPI across requests.

Usage:
  python -m shoptalk.rag.chain --query "red shirt for men under 50 dollars"
  python -m shoptalk.rag.chain --query "..." --stream
"""
import argparse
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from shoptalk.config import load_config
from shoptalk.rag.prompts import (
    build_prompt,
    format_catalog_block,
    format_history_block,
)
from shoptalk.retrieval.image_search import image_search as run_image_search
from shoptalk.retrieval.two_stage import two_stage_search

_llm = None
_chain = None


def _get_llm(cfg: dict):
    global _llm
    if _llm is None:
        lcfg = cfg["llm"]
        _llm = ChatOllama(
            model=lcfg["model"],
            base_url=lcfg["base_url"],
            temperature=lcfg["temperature"],
            num_predict=lcfg["num_predict"],
        )
    return _llm


def _get_chain(cfg: dict):
    global _chain
    if _chain is None:
        _chain = build_prompt() | _get_llm(cfg) | StrOutputParser()
    return _chain


def answer_from_hits(query: str, hits: list, history: list = None, cfg: dict = None, stream: bool = False):
    """Returns a full string (stream=False) or a generator of string chunks (stream=True)."""
    cfg = cfg or load_config()
    lcfg = cfg["llm"]
    chain = _get_chain(cfg)
    inputs = {
        "query": query,
        "catalog_block": format_catalog_block(hits[: lcfg["context_products"]]),
        "history_block": format_history_block(history or [], lcfg["conversation_max_turns"]),
    }
    return chain.stream(inputs) if stream else chain.invoke(inputs)


def rag_search_text(query: str, history: list = None, cfg: dict = None, stream: bool = False) -> dict:
    cfg = cfg or load_config()
    hits = two_stage_search(query, cfg=cfg)
    answer = answer_from_hits(query, hits, history, cfg=cfg, stream=stream)
    return {"hits": hits, "answer": answer}


def rag_search_image(image_path: str, history: list = None, cfg: dict = None, stream: bool = False) -> dict:
    """The LLM never sees the image -- it sees the BLIP pseudo-caption plus the
    (already visually-matched) retrieved product text, same as image_search's
    reranking step."""
    cfg = cfg or load_config()
    image_result = run_image_search(image_path, cfg=cfg)
    hits = image_result["hits"]
    pseudo_query = image_result["pseudo_query"]
    user_facing_query = (
        f"The user uploaded a photo. It looks like: {pseudo_query!r}. "
        f"Help them find this or a close match in the catalog."
    )
    answer = answer_from_hits(user_facing_query, hits, history, cfg=cfg, stream=stream)
    return {"hits": hits, "answer": answer, "pseudo_query": pseudo_query}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    result = rag_search_text(args.query, stream=args.stream)
    print(f"query: {args.query!r}")
    print(f"retrieved {len(result['hits'])} products\n")

    if args.stream:
        for chunk in result["answer"]:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
