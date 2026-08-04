"""
Semantic result caching (design doc §12.3: "semantic caching of frequent
queries -- embed query -> serve cached answer if cosine > threshold").

Two separate caches are exposed, not one, because the two things being
cached have different safety properties:

- retrieval cache: stage1 ANN + rerank hits. A pure function of the raw
  query text and the current catalog snapshot -- independent of
  conversation history -- so it's safe to reuse across ANY two requests
  with a near-duplicate query, regardless of session.
- full-response cache: the final LLM answer alongside the hits. The LLM
  prompt includes conversation history, so the same query text can
  legitimately deserve a different answer depending on what came before
  it in that session ("is there a cheaper option?" means nothing without
  history). This cache is only ever consulted/populated by main.py for
  requests with EMPTY history -- a fresh session's first turn -- the one
  case where "same query -> same answer" is actually true regardless of
  who's asking.

Threshold calibration (measured against the real bge-base-en-v1.5 query
encoder, see commit message for the full readout): true near-duplicates
("red phone case" vs "red phone cover") score ~0.94; but critically,
queries that must NOT share a cache entry because they differ only in a
number -- "red phone case under $30" vs "under $100" -- score ~0.86,
statistically indistinguishable from unrelated-but-topically-similar
queries ("looking for a red dress" vs "...size medium", also ~0.86).
Cosine similarity alone cannot separate a paraphrase from a changed
price/size/quantity constraint. So every lookup ALSO requires the two
queries' extracted numeric tokens to match exactly -- this closes that
gap without needing a much stricter (and therefore far less useful,
mostly-exact-match) cosine threshold.
"""
import re
import time

import numpy as np

from shoptalk.retrieval.search import BGE_QUERY_INSTRUCTION, _get_model

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def numeric_signature(text: str) -> tuple:
    return tuple(_NUMBER_RE.findall(text))


def embed_query(query: str, text_model_name: str) -> np.ndarray:
    """Same model + instruction prefix as the real stage1 retrieval path
    (shoptalk.retrieval.search.search) -- similarity scores here must live
    in the same embedding space as what actually drives retrieval, or a
    "cache hit" wouldn't mean the two queries are actually retrieval-
    equivalent."""
    model = _get_model(text_model_name)
    return model.encode([BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]


class _Entry:
    __slots__ = ("embedding", "numeric_sig", "payload", "expires_at")

    def __init__(self, embedding, numeric_sig, payload, expires_at):
        self.embedding = embedding
        self.numeric_sig = numeric_sig
        self.payload = payload
        self.expires_at = expires_at


class SemanticCache:
    """In-process cache -- no Redis in this stack, and a linear scan over a
    few hundred short embeddings is sub-millisecond, far cheaper than the
    multi-second retrieval/rerank (or 100+ second LLM) step it replaces on
    a hit. Bounded by both a TTL (catalog/prompt changes go stale) and a
    max entry count with FIFO eviction (bounded memory for one runaway
    process)."""

    def __init__(self, maxsize: int = 500, ttl_seconds: float = 600, similarity_threshold: float = 0.93):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list = []
        self.hits = 0
        self.misses = 0

    def _evict_expired(self, now: float):
        self._entries = [e for e in self._entries if e.expires_at > now]

    def get(self, query_text: str, embedding: np.ndarray):
        now = time.time()
        self._evict_expired(now)
        sig = numeric_signature(query_text)

        best_entry, best_sim = None, self.similarity_threshold
        for entry in self._entries:
            if entry.numeric_sig != sig:
                continue  # numeric-signature guard -- see module docstring
            sim = float(np.dot(entry.embedding, embedding))
            if sim >= best_sim:
                best_entry, best_sim = entry, sim

        if best_entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return best_entry.payload

    def put(self, query_text: str, embedding: np.ndarray, payload) -> None:
        now = time.time()
        self._evict_expired(now)
        self._entries.append(_Entry(embedding, numeric_signature(query_text), payload, now + self.ttl_seconds))
        if len(self._entries) > self.maxsize:
            self._entries.pop(0)  # oldest first (list append order == insertion order)

    def clear(self) -> None:
        self._entries.clear()
