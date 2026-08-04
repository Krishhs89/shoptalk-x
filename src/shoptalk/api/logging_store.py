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
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    history_json TEXT NOT NULL,   -- JSON list of {role, content}
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_name, updated_at);
"""

# Hard cap on stored turns per conversation -- the LLM prompt only ever uses
# the last `llm.conversation_max_turns` (default 6) anyway, so there's no
# quality reason to keep more; this just bounds worst-case storage growth
# for one runaway session.
MAX_STORED_TURNS = 200


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


def save_conversation(session_id: str, user_name: str, history: list) -> None:
    """Persists the full turn history for a conversation. Called after every
    turn (not just on session end) so a mid-conversation API restart never
    loses more than the single in-flight request. SQLite read/write of a
    small JSON blob is milliseconds -- negligible next to the LLM step it
    sits next to (seconds to minutes)."""
    bounded = history[-MAX_STORED_TURNS:]
    now = time.time()
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO conversations (session_id, user_name, history_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 history_json = excluded.history_json,
                 updated_at = excluded.updated_at""",
            (session_id, user_name, json.dumps(bounded), now, now),
        )
    conn.close()


def load_conversation(session_id: str) -> dict:
    """Returns {"user_name", "history", "updated_at"} or None if unknown --
    used both to resume the backend's context after a restart and to hydrate
    the UI when a user picks a past conversation from their list."""
    conn = _connect()
    row = conn.execute(
        "SELECT user_name, history_json, updated_at FROM conversations WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"user_name": row[0], "history": json.loads(row[1]), "updated_at": row[2]}


def list_conversations(user_name: str, limit: int = 20) -> list:
    """Lightweight list for a sidebar: session_id + a short preview + when it
    was last active + how many turns -- NOT the full history (that's a
    separate load_conversation call, only made when the user actually picks
    one, to avoid pulling every past conversation's full text just to render
    a list)."""
    conn = _connect()
    rows = conn.execute(
        """SELECT session_id, history_json, updated_at FROM conversations
           WHERE user_name = ? ORDER BY updated_at DESC LIMIT ?""",
        (user_name, limit),
    ).fetchall()
    conn.close()

    results = []
    for session_id, history_json, updated_at in rows:
        history = json.loads(history_json)
        first_user_turn = next((t["content"] for t in history if t["role"] == "user"), "(empty conversation)")
        preview = first_user_turn[:80] + ("..." if len(first_user_turn) > 80 else "")
        results.append(
            {"session_id": session_id, "preview": preview, "updated_at": updated_at, "turn_count": len(history)}
        )
    return results
