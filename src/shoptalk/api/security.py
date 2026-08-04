"""
Minimal security controls (design doc §8): API-key auth, per-client rate
limiting, request size caps. Deliberately dependency-light (no Redis/external
service) since this is a single-instance deployment -- swap the in-memory
rate-limit state for Redis if/when the service is scaled horizontally.
"""
import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

# `or None` matters: docker-compose's `${SHOPTALK_API_KEY:-}` substitution sets
# this to an empty string (not unset) when the host has no such env var, and
# os.environ.get() returns "" (not None) for a set-but-empty var -- without
# the `or None`, `"" is None` is False, so auth would wrongly enforce an
# empty-string key that no request could ever match. Unset OR empty -> disabled.
API_KEY = os.environ.get("SHOPTALK_API_KEY") or None
RATE_LIMIT_PER_MINUTE = int(os.environ.get("SHOPTALK_RATE_LIMIT_PER_MINUTE", "60"))
MAX_IMAGE_BYTES = int(os.environ.get("SHOPTALK_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10 MB

_request_log: dict = defaultdict(deque)  # client_key -> deque[timestamps]


def require_api_key(x_api_key: str = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled -- no key configured (local/dev mode)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def enforce_rate_limit(request: Request):
    client_key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window_start = now - 60
    log = _request_log[client_key]
    while log and log[0] < window_start:
        log.popleft()
    if len(log) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")
    log.append(now)


def check_image_size(content_length: int):
    if content_length > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )
