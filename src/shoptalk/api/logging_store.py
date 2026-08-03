"""
Prediction + feedback logging (design doc §6, "Prediction logging"): every
request's query, retrieved IDs, rerank scores, LLM output hash, and latency
breakdown, plus later user feedback keyed by request_id. SQLite for
simplicity at this scale; swap for Postgres/S3 parquet at higher volume
without changing the call sites (same three functions).
"""
import hashlib
import json
import sqlite3
import time

from shoptalk.config import REPO_ROOT

DB_PATH = REPO_ROOT / "data" / "logs" / "predictions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    endpoint TEXT NOT NULL,
    session_id TEXT,
    query TEXT,
    retrieved_ids TEXT,       -- JSON list
    rerank_scores TEXT,       -- JSON list
    llm_output_hash TEXT,
    stage1_ms REAL,
    rerank_ms REAL,
    llm_ms REAL,
    total_ms REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    FOREIGN KEY (request_id) REFERENCES predictions(request_id)
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def hash_output(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_prediction(
    request_id: str,
    endpoint: str,
    session_id: str,
    query: str,
    hits: list,
    llm_output: str,
    latency: dict,
) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO predictions
               (request_id, timestamp, endpoint, session_id, query, retrieved_ids,
                rerank_scores, llm_output_hash, stage1_ms, rerank_ms, llm_ms, total_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                time.time(),
                endpoint,
                session_id,
                query,
                json.dumps([h["item_id"] for h in hits]),
                json.dumps([round(h.get("rerank_score", 0.0), 4) for h in hits]),
                hash_output(llm_output),
                latency.get("stage1_ms"),
                latency.get("rerank_ms"),
                latency.get("llm_ms"),
                latency.get("total_ms"),
            ),
        )
    conn.close()


def log_feedback(request_id: str, rating: int, comment: str = None) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO feedback (request_id, timestamp, rating, comment) VALUES (?, ?, ?, ?)",
            (request_id, time.time(), rating, comment),
        )
    conn.close()
