import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.data.preprocess import clean_text, synthetic_price
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
