"""Unit tests for the SemanticCache primitive (src/shoptalk/api/semantic_cache.py)
using synthetic unit vectors instead of the real bi-encoder, so these run in
milliseconds. Covers the two properties the design hinges on: (1) a
near-duplicate query above the similarity threshold hits, an unrelated one
doesn't, and (2) the numeric-signature guard blocks a hit even at cosine
similarity 1.0 when the queries differ only by a number -- the exact failure
mode measured empirically against the real model (see main.py's cache
wiring commit message): "under $30" vs "under $100" scored ~0.86, no more
distinguishable by cosine alone than two genuinely different queries."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.api.semantic_cache import SemanticCache, numeric_signature


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_numeric_signature_extracts_numbers():
    assert numeric_signature("red phone case under $30") == ("30",)
    assert numeric_signature("no numbers here") == ()
    assert numeric_signature("between 10 and 20.5") == ("10", "20.5")


def test_exact_duplicate_query_hits():
    cache = SemanticCache(similarity_threshold=0.93)
    cache.put("red phone case", _unit([1.0, 0.0, 0.0]), {"answer": "cached"})
    assert cache.get("red phone case", _unit([1.0, 0.0, 0.0])) == {"answer": "cached"}
    assert cache.hits == 1 and cache.misses == 0


def test_near_duplicate_above_threshold_hits():
    cache = SemanticCache(similarity_threshold=0.93)
    cache.put("red phone case under $30", _unit([1.0, 0.0, 0.0]), {"answer": "cached"})
    # cosine(  [1,0,0], [0.99, 0.05, 0] normalized ) is well above 0.93
    result = cache.get("red phone cover under $30", _unit([0.99, 0.05, 0.0]))
    assert result == {"answer": "cached"}


def test_unrelated_query_below_threshold_misses():
    cache = SemanticCache(similarity_threshold=0.93)
    cache.put("red phone case under $30", _unit([1.0, 0.0, 0.0]), {"answer": "cached"})
    result = cache.get("blue laptop bag under $50", _unit([0.0, 1.0, 0.0]))
    assert result is None
    assert cache.misses == 1


def test_numeric_mismatch_blocks_hit_even_at_perfect_cosine_similarity():
    """The critical safety guard: same embedding direction (cosine == 1.0)
    must NOT be treated as a cache hit when the queries carry different
    numbers -- a shopping assistant's price/size/quantity constraints are
    exactly the kind of difference embeddings alone tend to blur."""
    cache = SemanticCache(similarity_threshold=0.93)
    cache.put("red phone case under $30", _unit([1.0, 0.0, 0.0]), {"answer": "$30 answer"})
    result = cache.get("red phone case under $100", _unit([1.0, 0.0, 0.0]))
    assert result is None


def test_ttl_expiry():
    cache = SemanticCache(ttl_seconds=0.01, similarity_threshold=0.93)
    cache.put("red phone case", _unit([1.0, 0.0, 0.0]), {"answer": "cached"})
    time.sleep(0.02)
    assert cache.get("red phone case", _unit([1.0, 0.0, 0.0])) is None


def test_maxsize_evicts_oldest_entry():
    cache = SemanticCache(maxsize=2, similarity_threshold=0.93)
    cache.put("query one #1", _unit([1.0, 0.0, 0.0]), "first")
    cache.put("query two #2", _unit([0.0, 1.0, 0.0]), "second")
    cache.put("query three #3", _unit([0.0, 0.0, 1.0]), "third")
    # the first entry should have been evicted (FIFO), the later two remain
    assert cache.get("query one #1", _unit([1.0, 0.0, 0.0])) is None
    assert cache.get("query three #3", _unit([0.0, 0.0, 1.0])) == "third"


def test_clear_empties_cache():
    cache = SemanticCache(similarity_threshold=0.93)
    cache.put("red phone case", _unit([1.0, 0.0, 0.0]), {"answer": "cached"})
    cache.clear()
    assert cache.get("red phone case", _unit([1.0, 0.0, 0.0])) is None


@pytest.mark.parametrize(
    "query_a, query_b, sim, should_hit",
    [
        ("looking for a red dress", "looking for a red dress size medium", 0.86, False),
        ("wireless earbuds under 20 dollars", "cheap wireless earbuds", 0.86, False),
        ("red phone case under $30", "red phone cover under $30", 0.94, True),
    ],
)
def test_calibrated_threshold_against_measured_real_model_scores(query_a, query_b, sim, should_hit):
    """Regression guard for the threshold choice itself: replays the actual
    cosine scores measured against BAAI/bge-base-en-v1.5 (see commit
    message) at the default threshold of 0.93, so a future threshold change
    can't silently flip these judgment calls without a test failing."""
    cache = SemanticCache(similarity_threshold=0.93)
    vec_a = _unit([1.0, 0.0])
    # construct vec_b at the measured angle from vec_a: cos(theta) = sim
    theta = np.arccos(np.clip(sim, -1.0, 1.0))
    vec_b = _unit([np.cos(theta), np.sin(theta)])
    cache.put(query_a, vec_a, "cached")
    result = cache.get(query_b, vec_b)
    assert (result is not None) == should_hit
