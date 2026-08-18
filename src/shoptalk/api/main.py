"""
FastAPI inference service (design doc §4.1). Every model is loaded exactly
once, at startup, via the lifespan handler -- never per-request.

Endpoints:
  POST /search/text   text query -> RAG answer + product hits (SSE if stream=true)
  POST /search/image  photo upload -> RAG answer + product hits
  POST /verify         order photo verification (Day 5's Siamese/MLP head)
  POST /verify/quantity quantity/count check against a claimed count (pretrained YOLO)
  POST /feedback        thumbs up/down on a prior request_id
  GET  /health
  GET  /metrics         Prometheus exposition format
"""
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from shoptalk.api import logging_store
from shoptalk.api.schemas import (
    ConversationDetailResponse,
    ConversationListResponse,
    FeedbackRequest,
    HealthResponse,
    ImageSearchResponse,
    LatencyBreakdown,
    ProductHit,
    QuantityCheckResponse,
    SearchResponse,
    TextSearchRequest,
    VerifyResponse,
)
from shoptalk.api.security import check_image_size, enforce_rate_limit, require_api_key
from shoptalk.api.semantic_cache import SemanticCache, embed_query
from shoptalk.config import load_config
from shoptalk.rag.chain import _get_chain, _get_llm, answer_from_hits
from shoptalk.rag.prompts import format_catalog_block, format_history_block
from shoptalk.retrieval.image_search import (
    _get_blip,
    _get_clip,
    caption_query_image,
    clip_ann_search,
)
from shoptalk.retrieval.rerank import _get_cross_encoder, rerank
from shoptalk.retrieval.search import _get_collection as _get_text_collection
from shoptalk.retrieval.search import _get_model as _get_text_model
from shoptalk.retrieval.search import search as stage1_search

REQUEST_COUNT = Counter("shoptalk_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("shoptalk_request_latency_seconds", "End-to-end request latency", ["endpoint"])
STAGE_LATENCY = Histogram("shoptalk_stage_latency_seconds", "Per-stage latency", ["endpoint", "stage"])
CACHE_HIT_COUNT = Counter("shoptalk_cache_hits_total", "Semantic cache hits", ["cache"])
CACHE_MISS_COUNT = Counter("shoptalk_cache_misses_total", "Semantic cache misses", ["cache"])

# In-process cache of session_id -> [{"role", "content"}, ...], backed by the
# SQLite `conversations` table (logging_store) for durability across restarts
# and for the UI's history/resume feature. The cache avoids a DB round trip
# on every token of a streamed response; every completed turn is written
# through to SQLite so a restart never loses more than one in-flight turn.
_state = {"cfg": None, "sessions": {}, "session_users": {}}

# Two semantic caches (see api/semantic_cache.py for why they're separate),
# built lazily on first request so tests can spin up the app without a real
# cfg. "retrieval" caches stage1+rerank hits (safe for any query, any
# session); "full_response" caches the complete LLM answer too, but is only
# ever consulted for a session's first turn -- see search_text() below.
_caches = {"retrieval": None, "full_response": None}


def _get_cache(name: str, cfg: dict) -> SemanticCache:
    if _caches[name] is None:
        ccfg = cfg.get("cache", {})
        _caches[name] = SemanticCache(
            maxsize=ccfg.get("max_entries", 500),
            ttl_seconds=ccfg.get("ttl_seconds", 600),
            similarity_threshold=ccfg.get("similarity_threshold", 0.93),
        )
    return _caches[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["cfg"] = cfg
    print("warming models at startup...")
    _get_text_model(cfg["embeddings"]["text_model"])
    _get_text_collection(cfg["embeddings"]["chroma_dir"], cfg["embeddings"]["collection_name"])
    _get_cross_encoder(cfg["retrieval"]["rerank_model"])
    _get_clip(cfg["clip"])
    _get_blip(cfg["captioning"])
    _get_llm(cfg)
    print("all models warmed; service ready")
    yield


API_DESCRIPTION = """
Conversational shopping assistant over a ~10k-product catalog: natural-language
and photo search grounded by a two-stage retrieval pipeline (bi-encoder ANN +
cross-encoder rerank) and an LLM (Llama 3.1 8B via Ollama), plus order
verification (does the delivered photo match what was ordered?) and quantity
verification (does the delivered *count* match what was claimed?).

Every response that carries model inference reports its own latency, and every
verification-style endpoint returns a third **`suspect`** verdict alongside
`match`/`mismatch` for anything close to the model's decision boundary —
routed to human review, never resolved automatically.

Full write-up: [PROJECT_DEFINITION.md](https://github.com/Krishhs89/shoptalk-x/blob/master/docs/PROJECT_DEFINITION.md)
"""

TAGS_METADATA = [
    {
        "name": "Search",
        "description": "Natural-language and photo product search, backed by two-stage retrieval + RAG.",
    },
    {
        "name": "Verification",
        "description": "Does the delivered item match what was ordered — and how many arrived?",
    },
    {"name": "Feedback", "description": "Explicit thumbs up/down on a prior response, for the retraining loop."},
    {"name": "Conversations", "description": "List and resume a user's past conversation history."},
    {"name": "System", "description": "Liveness and Prometheus metrics."},
]

app = FastAPI(
    title="ShopTalk-X API",
    description=API_DESCRIPTION,
    version="1.0.0",
    contact={"name": "ShopTalk-X", "url": "https://github.com/Krishhs89/shoptalk-x"},
    license_info={"name": "See repository for license terms"},
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)


def _hits_to_products(hits: list) -> list:
    return [
        ProductHit(
            item_id=h["item_id"],
            item_name=h["metadata"]["item_name"],
            category=h["metadata"]["category"],
            brand=h["metadata"]["brand"],
            price_usd=h["metadata"]["price_usd"],
            stage1_score=h.get("stage1_score", 0.0),
            rerank_score=h.get("rerank_score", 0.0),
        )
        for h in hits
    ]


def _get_session(session_id: str, user_name: str = None) -> list:
    if session_id not in _state["sessions"]:
        persisted = logging_store.load_conversation(session_id)
        if persisted:
            _state["sessions"][session_id] = persisted["history"]
            _state["session_users"].setdefault(session_id, persisted["user_name"])
        else:
            _state["sessions"][session_id] = []
    if user_name:
        _state["session_users"][session_id] = user_name
    return _state["sessions"][session_id]


@app.middleware("http")
async def latency_and_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNT.labels(endpoint=endpoint, status=str(response.status_code)).inc()
    response.headers["X-Latency-Ms"] = f"{duration * 1000:.1f}"
    return response


@app.get(
    "/health", response_model=HealthResponse,
    tags=["System"], summary="Liveness + which models are actually loaded",
)
def health():
    cfg = _state["cfg"]
    # The in-process models (embedding/reranker/CLIP/BLIP) are warmed at
    # startup and the app wouldn't be serving requests if that failed, so
    # "loaded" is accurate for them. Ollama is a separate process/container
    # that can go down *after* startup (e.g. restarted, OOM-killed) without
    # this process knowing -- a static "ok" here would keep telling the UI
    # everything's fine while every search silently fails at the LLM step.
    # A cheap live probe against Ollama's own list-models endpoint catches
    # that case.
    ollama_ok = False
    try:
        resp = requests.get(f"{cfg['llm']['base_url']}/api/tags", timeout=2)
        ollama_ok = resp.ok
    except requests.RequestException:
        ollama_ok = False

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        models_loaded={
            "text_embedding": cfg["embeddings"]["text_model"],
            "reranker": cfg["retrieval"]["rerank_model"],
            "clip": f"{cfg['clip']['model_name']}/{cfg['clip']['pretrained']}",
            "captioner": cfg["captioning"]["model"],
            "llm": cfg["llm"]["model"] + ("" if ollama_ok else " (UNREACHABLE)"),
        },
    )


@app.get("/metrics", tags=["System"], summary="Prometheus exposition format")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _respond_or_stream(
    endpoint: str,
    query_for_llm: str,
    reranked: list,
    session_id: str,
    stage1_ms: float,
    rerank_ms: float,
    stream: bool,
    user_name: str = None,
    cached_answer: str = None,
    extra_fields: dict = None,
):
    cfg = _state["cfg"]
    history = _get_session(session_id, user_name)
    request_id = str(uuid.uuid4())

    if stream:
        def token_stream():
            chain = _get_chain(cfg)
            inputs = {
                "query": query_for_llm,
                "catalog_block": format_catalog_block(reranked[: cfg["llm"]["context_products"]]),
                "history_block": format_history_block(history, cfg["llm"]["conversation_max_turns"]),
            }
            chunks = []
            t_llm0 = time.perf_counter()
            for chunk in chain.stream(inputs):
                chunks.append(chunk)
                yield f"data: {chunk}\n\n"
            llm_ms = (time.perf_counter() - t_llm0) * 1000
            STAGE_LATENCY.labels(endpoint=endpoint, stage="llm").observe(llm_ms / 1000)

            answer_text = "".join(chunks)
            history.append({"role": "user", "content": query_for_llm})
            history.append({"role": "assistant", "content": answer_text})
            logging_store.log_prediction(
                request_id, endpoint, session_id, query_for_llm, reranked, answer_text,
                {"stage1_ms": stage1_ms, "rerank_ms": rerank_ms, "llm_ms": llm_ms,
                 "total_ms": stage1_ms + rerank_ms + llm_ms},
            )
            logging_store.save_conversation(
                session_id, _state["session_users"].get(session_id) or "anonymous", history
            )
            yield f"event: done\ndata: {request_id}\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    if cached_answer is not None:
        # Full-response semantic cache hit (see search_text) -- the LLM step
        # is skipped entirely, not just made faster.
        answer_text = cached_answer
        llm_ms = 0.0
    else:
        t_llm0 = time.perf_counter()
        answer_text = answer_from_hits(query_for_llm, reranked, history, cfg=cfg, stream=False)
        llm_ms = (time.perf_counter() - t_llm0) * 1000
        STAGE_LATENCY.labels(endpoint=endpoint, stage="llm").observe(llm_ms / 1000)

    history.append({"role": "user", "content": query_for_llm})
    history.append({"role": "assistant", "content": answer_text})

    latency = LatencyBreakdown(
        stage1_ms=stage1_ms, rerank_ms=rerank_ms, llm_ms=llm_ms,
        total_ms=stage1_ms + rerank_ms + llm_ms,
    )
    logging_store.log_prediction(
        request_id, endpoint, session_id, query_for_llm, reranked, answer_text, latency.model_dump()
    )
    logging_store.save_conversation(
        session_id, _state["session_users"].get(session_id) or "anonymous", history
    )

    fields = dict(
        answer=answer_text, hits=_hits_to_products(reranked), latency=latency,
        session_id=session_id, request_id=request_id,
    )
    fields.update(extra_fields or {})
    response_cls = ImageSearchResponse if extra_fields else SearchResponse
    return response_cls(**fields)


@app.post(
    "/search/text", response_model=SearchResponse, dependencies=[Depends(require_api_key)],
    tags=["Search"], summary="Natural-language product search",
)
def search_text(req: TextSearchRequest, request: Request):
    enforce_rate_limit(request)
    cfg = _state["cfg"]
    session_id = req.session_id or str(uuid.uuid4())

    # Semantic caching (design doc §12.3) is skipped for streamed responses
    # -- a cache hit means "return the whole answer instantly", which is a
    # different response shape (JSON, not an SSE token stream) than a
    # client asking to stream expects.
    cache_enabled = cfg.get("cache", {}).get("enabled", True) and not req.stream
    # "Fresh" = no conversation history yet for this session. Only fresh
    # turns are eligible for the full-response cache: the cached answer was
    # generated with no prior context, so it's only valid to hand back to
    # another request that also has no prior context (see semantic_cache.py
    # docstring for why the LLM answer can't be cached across sessions with
    # different history).
    is_fresh_turn = cache_enabled and not _get_session(session_id)
    query_embedding = embed_query(req.query, cfg["embeddings"]["text_model"]) if cache_enabled else None

    if is_fresh_turn:
        full_cache = _get_cache("full_response", cfg)
        cached = full_cache.get(req.query, query_embedding)
        if cached is not None:
            CACHE_HIT_COUNT.labels(cache="full_response").inc()
            return _respond_or_stream(
                "/search/text", req.query, cached["hits"], session_id, 0.0, 0.0, False,
                user_name=req.user_name, cached_answer=cached["answer"],
            )
        CACHE_MISS_COUNT.labels(cache="full_response").inc()

    retrieval_cache = _get_cache("retrieval", cfg) if cache_enabled else None
    cached_reranked = retrieval_cache.get(req.query, query_embedding) if retrieval_cache else None

    if cached_reranked is not None:
        CACHE_HIT_COUNT.labels(cache="retrieval").inc()
        reranked = cached_reranked
        stage1_ms = rerank_ms = 0.0
    else:
        if retrieval_cache is not None:
            CACHE_MISS_COUNT.labels(cache="retrieval").inc()
        t0 = time.perf_counter()
        hits = stage1_search(req.query, top_k=cfg["retrieval"]["stage1_k"], cfg=cfg)
        t1 = time.perf_counter()
        reranked = rerank(req.query, hits, top_k=req.top_k or cfg["retrieval"]["top_k"], cfg=cfg)
        t2 = time.perf_counter()
        stage1_ms, rerank_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        STAGE_LATENCY.labels(endpoint="/search/text", stage="stage1").observe(t1 - t0)
        STAGE_LATENCY.labels(endpoint="/search/text", stage="rerank").observe(t2 - t1)
        if retrieval_cache is not None:
            retrieval_cache.put(req.query, query_embedding, reranked)

    response = _respond_or_stream(
        "/search/text", req.query, reranked, session_id, stage1_ms, rerank_ms, req.stream,
        user_name=req.user_name,
    )

    if is_fresh_turn and not req.stream:
        _get_cache("full_response", cfg).put(
            req.query, query_embedding, {"answer": response.answer, "hits": reranked}
        )

    return response


@app.post(
    "/search/image", response_model=ImageSearchResponse, dependencies=[Depends(require_api_key)],
    tags=["Search"], summary="Photo-based product search",
)
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = None,
    user_name: str = None,
    top_k: int = None,
    stream: bool = False,
):
    enforce_rate_limit(request)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="file must be an image")

    content = await file.read()
    check_image_size(len(content))

    cfg = _state["cfg"]
    session_id = session_id or str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "upload.jpg").suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        candidates = clip_ann_search(tmp_path, cfg["retrieval"]["image_stage1_k"], cfg)
        t1 = time.perf_counter()
        pseudo_query = caption_query_image(tmp_path, cfg)
        reranked = rerank(pseudo_query, candidates, top_k=top_k or cfg["retrieval"]["top_k"], cfg=cfg)
        t2 = time.perf_counter()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    stage1_ms, rerank_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
    STAGE_LATENCY.labels(endpoint="/search/image", stage="stage1").observe(t1 - t0)
    STAGE_LATENCY.labels(endpoint="/search/image", stage="rerank").observe(t2 - t1)

    query_for_llm = (
        f"The user uploaded a photo. It looks like: {pseudo_query!r}. "
        f"Help them find this or a close match in the catalog."
    )
    return _respond_or_stream(
        "/search/image", query_for_llm, reranked, session_id, stage1_ms, rerank_ms, stream,
        user_name=user_name, extra_fields={"pseudo_query": pseudo_query},
    )


@app.post(
    "/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key)],
    tags=["Verification"], summary="Does the delivered photo match the ordered item?",
)
async def verify(request: Request, order_item_id: str, file: UploadFile = File(...)):
    enforce_rate_limit(request)
    content = await file.read()
    check_image_size(len(content))

    try:
        from shoptalk.verification.verify import verify_photo
    except ImportError:
        raise HTTPException(status_code=501, detail="verification head not trained yet -- see Day 5")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        t0 = time.perf_counter()
        result = verify_photo(tmp_path, order_item_id, cfg=_state["cfg"])
        latency_ms = (time.perf_counter() - t0) * 1000
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return VerifyResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        order_item_id=order_item_id,
        request_id=str(uuid.uuid4()),
        latency_ms=latency_ms,
    )


@app.post(
    "/verify/quantity", response_model=QuantityCheckResponse, dependencies=[Depends(require_api_key)],
    tags=["Verification"], summary="Does the delivered count match the claimed quantity?",
)
async def verify_quantity_endpoint(
    request: Request, order_item_id: str, claimed_qty: int, file: UploadFile = File(...)
):
    enforce_rate_limit(request)
    content = await file.read()
    check_image_size(len(content))

    try:
        from shoptalk.counting.count import verify_quantity
    except ImportError:
        raise HTTPException(
            status_code=501, detail="quantity validation not available -- see docs/model_cards/quantity_counting.md"
        )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        t0 = time.perf_counter()
        result = verify_quantity(tmp_path, order_item_id, claimed_qty, cfg=_state["cfg"])
        latency_ms = (time.perf_counter() - t0) * 1000
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return QuantityCheckResponse(
        verdict=result["verdict"],
        claimed_qty=claimed_qty,
        detected_count=result.get("detected_count"),
        matched_class=result.get("matched_class"),
        message=result.get("message"),
        order_item_id=order_item_id,
        request_id=str(uuid.uuid4()),
        latency_ms=latency_ms,
    )


@app.post("/feedback", tags=["Feedback"], summary="Thumbs up/down on a prior response")
def feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating must be +1 or -1")
    logging_store.log_feedback(req.request_id, req.rating, req.comment)
    return {"status": "recorded"}


@app.get(
    "/conversations", response_model=ConversationListResponse, dependencies=[Depends(require_api_key)],
    tags=["Conversations"], summary="List a user's past conversations",
)
def list_conversations(user_name: str, limit: int = 20):
    if not user_name.strip():
        raise HTTPException(status_code=422, detail="user_name is required")
    return ConversationListResponse(conversations=logging_store.list_conversations(user_name, limit))


@app.get(
    "/conversations/{session_id}",
    response_model=ConversationDetailResponse,
    dependencies=[Depends(require_api_key)],
    tags=["Conversations"],
    summary="Resume a specific past conversation",
)
def get_conversation(session_id: str):
    convo = logging_store.load_conversation(session_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationDetailResponse(
        session_id=session_id,
        user_name=convo["user_name"],
        history=convo["history"],
        updated_at=convo["updated_at"],
    )
