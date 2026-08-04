"""Guards against a real bug seen live: st.markdown() treats paired $...$ as
inline LaTeX, so any assistant answer mentioning two or more prices (very
common for a shopping assistant) had the text between them silently parsed
as a math formula instead of displayed -- "LLM text output isn't showing up
right." Escaping $ before rendering fixes it while leaving other markdown
(bold, lists) intact."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.ui.app import escape_markdown_dollars


def test_single_price_untouched_in_content():
    text = "I recommend the phone case priced at $36.42."
    escaped = escape_markdown_dollars(text)
    assert "36.42" in escaped
    assert "\\$36.42" in escaped


def test_two_prices_dont_form_a_latex_pair():
    """Before the fix, everything between the first and second $ here would
    be parsed as a LaTeX formula by Streamlit instead of shown as text."""
    text = "Option A is $20.00 and Option B is $30.00."
    escaped = escape_markdown_dollars(text)
    # no bare (unescaped) $ remains that could pair up as math delimiters
    assert "$" not in escaped.replace("\\$", "")
    assert escaped.count("\\$") == 2


def test_other_markdown_formatting_preserved():
    text = "**Bold recommendation:** this one is great, priced at $19.99."
    escaped = escape_markdown_dollars(text)
    assert escaped.startswith("**Bold recommendation:**")
    assert "\\$19.99" in escaped


def test_text_without_dollar_signs_unchanged():
    text = "This is a great colorful phone case with a snug fit."
    assert escape_markdown_dollars(text) == text
