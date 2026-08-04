import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.data.preprocess import clean_text, drop_duplicate_item_ids, synthetic_price
from shoptalk.eval.generate_golden_set import (
    is_clean_attribute,
    price_threshold,
    readable_category,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\nc") == "a b c"


def test_clean_text_strips_zero_width_space():
    assert clean_text("hello\u200bworld") == "hello world"


def test_synthetic_price_is_deterministic():
    p1 = synthetic_price("B0TEST123", "SHOES")
    p2 = synthetic_price("B0TEST123", "SHOES")
    assert p1 == p2


def test_synthetic_price_within_category_band():
    price = synthetic_price("B0TEST123", "SHOES")
    assert 25 <= price <= 150  # SHOES band from CATEGORY_PRICE_BANDS


def test_synthetic_price_uses_default_band_for_unknown_category():
    price = synthetic_price("B0TEST999", "SOME_UNKNOWN_CATEGORY")
    assert 10 <= price <= 200  # DEFAULT_PRICE_BAND


def test_is_clean_attribute_rejects_leaked_codes():
    assert is_clean_attribute("#lightness") is False


def test_is_clean_attribute_rejects_mixed_script():
    assert is_clean_attribute("AJC Collection(AJC コレクション)") is False


def test_is_clean_attribute_accepts_normal_brand():
    assert is_clean_attribute("Amazon Brand - Solimo") is True


def test_is_clean_attribute_rejects_empty():
    assert is_clean_attribute("") is False


def test_price_threshold_exceeds_actual_price():
    assert price_threshold(42.0) > 42.0


def test_readable_category_formats_underscores():
    assert readable_category("CELLULAR_PHONE_CASE") == "cellular phone case"


def test_drop_duplicate_item_ids_keeps_first_occurrence():
    """Real bug found at 10k-product Colab scale: the same item_id can appear
    twice across ABO's listing shards (two listing entries for one ASIN),
    and every downstream consumer assumes item_id is unique -- Chroma's
    collection.add(ids=...) crashed with DuplicateIDError, and build_pairs.py
    crashed with a pandas "truth value of a Series is ambiguous" error from
    .get() on a non-unique index. Both traced back to this."""
    df = pd.DataFrame(
        {
            "item_id": ["A1", "A2", "A1", "A3"],
            "item_name": ["first A1 listing", "A2", "second A1 listing", "A3"],
        }
    )
    deduped, duplicate_count = drop_duplicate_item_ids(df)
    assert duplicate_count == 1
    assert list(deduped["item_id"]) == ["A1", "A2", "A3"]
    assert deduped.loc[deduped["item_id"] == "A1", "item_name"].iloc[0] == "first A1 listing"


def test_drop_duplicate_item_ids_no_duplicates_is_a_no_op():
    df = pd.DataFrame({"item_id": ["A1", "A2"], "item_name": ["a", "b"]})
    deduped, duplicate_count = drop_duplicate_item_ids(df)
    assert duplicate_count == 0
    assert len(deduped) == 2
