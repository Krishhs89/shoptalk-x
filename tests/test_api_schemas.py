"""Guards against PEP 604 (`X | None`) union syntax creeping into pydantic
models -- it silently breaks class definition on Python < 3.10 (caught this
for real: a ruff autofix converted Optional[str] to str | None and the API
failed to import under the project's Python 3.9 dev venv)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoptalk.api.schemas import FeedbackRequest, TextSearchRequest, VerifyResponse


def test_text_search_request_defaults():
    req = TextSearchRequest(query="red shirt")
    assert req.session_id is None
    assert req.top_k is None
    assert req.stream is False


def test_feedback_request_optional_comment():
    req = FeedbackRequest(request_id="abc", rating=1)
    assert req.comment is None


def test_verify_response_roundtrip():
    resp = VerifyResponse(verdict="match", confidence=0.9, threshold=0.5, order_item_id="B01", request_id="r1")
    assert resp.verdict == "match"
