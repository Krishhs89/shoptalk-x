"""Pydantic request/response models for the FastAPI service."""

from typing import Optional

from pydantic import BaseModel, Field


class ProductHit(BaseModel):
    item_id: str
    item_name: str
    category: str
    brand: str
    price_usd: float
    stage1_score: float
    rerank_score: float


class LatencyBreakdown(BaseModel):
    stage1_ms: float
    rerank_ms: float
    llm_ms: float
    total_ms: float


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, description="reused across turns for follow-up support")
    user_name: Optional[str] = Field(None, description="associates this conversation with a user for history/resume")
    top_k: Optional[int] = Field(None, ge=1, le=50)
    stream: bool = False


class SearchResponse(BaseModel):
    answer: str
    hits: list[ProductHit]
    latency: LatencyBreakdown
    session_id: str
    request_id: str


class ImageSearchResponse(SearchResponse):
    pseudo_query: str


class VerifyRequest(BaseModel):
    order_item_id: str = Field(..., description="item_id of the ordered product")


class VerifyResponse(BaseModel):
    verdict: str  # "match" | "mismatch" | "suspect"
    confidence: float
    threshold: float
    order_item_id: str
    request_id: str


class QuantityCheckRequest(BaseModel):
    order_item_id: str = Field(..., description="item_id of the ordered product")
    claimed_qty: int = Field(..., ge=1, le=100, description="quantity the customer claims to have received")


class QuantityCheckResponse(BaseModel):
    verdict: str  # "match" | "mismatch" | "suspect" | "unsupported"
    claimed_qty: int
    detected_count: Optional[int] = None
    matched_class: Optional[str] = None
    message: Optional[str] = None
    order_item_id: str
    request_id: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(..., description="+1 (thumbs up) or -1 (thumbs down)")
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict


class ConversationSummary(BaseModel):
    session_id: str
    preview: str
    updated_at: float
    turn_count: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class ConversationTurn(BaseModel):
    role: str
    content: str


class ConversationDetailResponse(BaseModel):
    session_id: str
    user_name: str
    history: list[ConversationTurn]
    updated_at: float
