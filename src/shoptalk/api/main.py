"""
FastAPI inference service (design doc §4.1). Every model is loaded exactly
once, at startup, via the lifespan handler -- never per-request.

Endpoints:
  POST /search/text   text query -> RAG answer + product hits (SSE if stream=true)
  POST /search/image  photo upload -> RAG answer + product hits
  POST /verify         order photo verification (Day 5's Siamese/MLP head)
  POST /feedback        thumbs up/down on a prior request_id
  GET  /health
  GET  /metrics         Prometheus exposition format
"""
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from shoptalk.api import logging_store
from shoptalk.api.schemas import (
    FeedbackRequest,
    HealthResponse,
    ImageSearchResponse,
    LatencyBreakdown,
    ProductHit,
    SearchResponse,
    TextSearchRequest,
    VerifyResponse,
)
from shoptalk.api.security import check_image_size, enforce_rate_limit, require_api_key
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

# In-memory session store: session_id -> [{"role", "content"}, ...]. Fine for a
# single-instance dev/demo deployment; swap for Redis before scaling out.
_state = {"cfg": None, "sessions": {}}


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


app = FastAPI(title="ShopTalk-X API", version="0.1.0", lifespan=lifespan)


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


def _get_session(session_id: str) -> list:
    return _state["sessions"].setdefault(session_id, [])


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


@app.get("/health", response_model=HealthResponse)
def health():
    cfg = _state["cfg"]
    return HealthResponse(
        status="ok",
        models_loaded={
            "text_embedding": cfg["embeddings"]["text_model"],
            "reranker": cfg["retrieval"]["rerank_model"],
            "clip": f"{cfg['clip']['model_name']}/{cfg['clip']['pretrained']}",
            "captioner": cfg["captioning"]["model"],
            "llm": cfg["llm"]["model"],
        },
    )


@app.get("/metrics")
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
    extra_fields: dict = None,
):
    cfg = _state["cfg"]
    history = _get_session(session_id)
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
            yield f"event: done\ndata: {request_id}\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

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

    fields = dict(
        answer=answer_text, hits=_hits_to_products(reranked), latency=latency,
        session_id=session_id, request_id=request_id,
    )
    fields.update(extra_fields or {})
    response_cls = ImageSearchResponse if extra_fields else SearchResponse
    return response_cls(**fields)


@app.post("/search/text", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search_text(req: TextSearchRequest, request: Request):
    enforce_rate_limit(request)
    cfg = _state["cfg"]
    session_id = req.session_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    hits = stage1_search(req.query, top_k=cfg["retrieval"]["stage1_k"], cfg=cfg)
    t1 = time.perf_counter()
    reranked = rerank(req.query, hits, top_k=req.top_k or cfg["retrieval"]["top_k"], cfg=cfg)
    t2 = time.perf_counter()

    stage1_ms, rerank_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
    STAGE_LATENCY.labels(endpoint="/search/text", stage="stage1").observe(t1 - t0)
    STAGE_LATENCY.labels(endpoint="/search/text", stage="rerank").observe(t2 - t1)

    return _respond_or_stream(
        "/search/text", req.query, reranked, session_id, stage1_ms, rerank_ms, req.stream
    )


@app.post("/search/image", response_model=ImageSearchResponse, dependencies=[Depends(require_api_key)])
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = None,
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
        extra_fields={"pseudo_query": pseudo_query},
    )


@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key)])
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
        result = verify_photo(tmp_path, order_item_id, cfg=_state["cfg"])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return VerifyResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        order_item_id=order_item_id,
        request_id=str(uuid.uuid4()),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating must be +1 or -1")
    logging_store.log_feedback(req.request_id, req.rating, req.comment)
    return {"status": "recorded"}
