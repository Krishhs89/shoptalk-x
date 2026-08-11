import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.counting.coco_classes import resolve_coco_class
from shoptalk.counting.count import _compute_verdict


def test_resolve_coco_class_direct_category_match():
    assert resolve_coco_class("Acme Office Chair", "CHAIR") == "chair"


def test_resolve_coco_class_keyword_match_in_name():
    assert resolve_coco_class("Solimo 500ml Water Bottle Pack of 6", "GROCERY") == "bottle"


def test_resolve_coco_class_keyword_beats_no_category_match():
    assert resolve_coco_class("Ceramic Vase Set", "HOME_FURNITURE_AND_DECOR") == "vase"


def test_resolve_coco_class_no_match_returns_none():
    assert resolve_coco_class("Amazon Brand Mobile Cover for Galaxy A50s", "CELLULAR_PHONE_CASE") is None


def test_compute_verdict_exact_match():
    assert _compute_verdict(detected_count=3, claimed_qty=3, suspect_margin=1) == "match"


def test_compute_verdict_within_margin_is_suspect():
    assert _compute_verdict(detected_count=2, claimed_qty=3, suspect_margin=1) == "suspect"
    assert _compute_verdict(detected_count=4, claimed_qty=3, suspect_margin=1) == "suspect"


def test_compute_verdict_beyond_margin_is_mismatch():
    assert _compute_verdict(detected_count=0, claimed_qty=3, suspect_margin=1) == "mismatch"
    assert _compute_verdict(detected_count=6, claimed_qty=3, suspect_margin=1) == "mismatch"


def test_compute_verdict_zero_margin_requires_exact():
    assert _compute_verdict(detected_count=2, claimed_qty=3, suspect_margin=0) == "mismatch"
