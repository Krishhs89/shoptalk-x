import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.rag.prompts import format_catalog_block, format_history_block


def _hit(item_id="B01", name="Red Shirt", category="SHIRT", brand="Acme", price=19.99, doc="a red shirt"):
    return {
        "item_id": item_id,
        "document": doc,
        "metadata": {"item_name": name, "category": category, "brand": brand, "price_usd": price},
    }


def test_format_catalog_block_empty():
    assert "no matching products" in format_catalog_block([])


def test_format_catalog_block_includes_item_id_and_price():
    block = format_catalog_block([_hit()])
    assert 'item_id="B01"' in block
    assert "19.99" in block


def test_format_catalog_block_delimits_each_product():
    block = format_catalog_block([_hit("B01"), _hit("B02")])
    assert block.count("<product") == 2
    assert block.count("</product>") == 2


def test_format_history_block_empty_history():
    assert format_history_block([], max_turns=5) == ""


def test_format_history_block_labels_speakers():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    block = format_history_block(history, max_turns=5)
    assert "User: hi" in block
    assert "Assistant: hello" in block


def test_format_history_block_respects_max_turns():
    history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    block = format_history_block(history, max_turns=2)
    # max_turns=2 keeps the last 2*2=4 entries
    assert "msg9" in block
    assert "msg0" not in block
