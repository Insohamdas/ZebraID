"""
zebraid/federation/org_service.py
⭐ NOVEL — OrgShard: FastAPI service representing one conservation organization.

Each OrgShard:
  - Owns a local ZebraIndex (FAISS + SQLite metadata).
  - Exposes a /match endpoint that accepts a Z-Hash (bytes) and returns
    ONLY a match confidence bucket and an opaque record_id.
    → No raw image, no GPS coordinates, no precise similarity score ever leaves.
  - Logs every inbound query for the privacy audit.
  - Enforces API key authentication and per-minute rate limiting.

Run two instances for the federated demo:
    uvicorn zebraid.federation.org_service:create_app --factory \
        --port 8001 --env-file .env.orga
    uvicorn zebraid.federation.org_service:create_app --factory \
        --port 8002 --env-file .env.orgb
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from zebraid.matching.index import ZebraIndex, MatchStatus


# ── Score quantization ────────────────────────────────────────────────────────
# Maps similarity score to coarse bucket — reduces information leakage per query.
SCORE_BUCKET_NO_MATCH      = "NO_MATCH"
SCORE_BUCKET_POSSIBLE      = "POSSIBLE_MATCH"
SCORE_BUCKET_STRONG        = "STRONG_MATCH"

THRESHOLD_POSSIBLE = 0.50   # similarity ≥ this → POSSIBLE_MATCH
THRESHOLD_STRONG   = 0.75   # similarity ≥ this → STRONG_MATCH


def _quantize_score(score: float) -> str:
    if score >= THRESHOLD_STRONG:
        return SCORE_BUCKET_STRONG
    elif score >= THRESHOLD_POSSIBLE:
        return SCORE_BUCKET_POSSIBLE
    else:
        return SCORE_BUCKET_NO_MATCH


# ── Request / Response models ─────────────────────────────────────────────────

class MatchRequest(BaseModel):
    z_hash_hex: str           # Z-Hash encoded as hex string (no raw image, no GPS)
    requester_org_id: str     # which org is querying (for audit log)


class MatchResponse(BaseModel):
    match_bucket: str         # NO_MATCH | POSSIBLE_MATCH | STRONG_MATCH
    record_id: Optional[str]  # opaque ID — no individual name, no GPS
    top_k_buckets: list[str]  # buckets for top-k candidates (no raw scores)
    query_id: str             # for cross-referencing with audit log


# ── Audit log schema ──────────────────────────────────────────────────────────
AUDIT_DB_PATH = os.environ.get("AUDIT_DB_PATH", "results/audit_log.sqlite")


def _get_audit_db() -> sqlite3.Connection:
    Path(AUDIT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(AUDIT_DB_PATH, check_same_thread=False)
    db.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            query_id        TEXT PRIMARY KEY,
            timestamp_utc   TEXT,
            requester_org   TEXT,
            this_org        TEXT,
            z_hash_bytes    INTEGER,
            response_bucket TEXT,
            record_id_returned TEXT,
            raw_image_sent  INTEGER DEFAULT 0,
            gps_sent        INTEGER DEFAULT 0
        )
    """)
    db.commit()
    return db


# ── Auth ──────────────────────────────────────────────────────────────────────
VALID_API_KEYS: set[str] = set(
    os.environ.get("ZEBRAID_API_KEYS", "demo-key-orga,demo-key-orgb").split(",")
)


def _verify_api_key(request: Request) -> str:
    key = request.headers.get("X-API-Key", "")
    if key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Rate limiter (in-memory, per requester) ───────────────────────────────────
_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))


def _check_rate_limit(requester: str) -> None:
    now = time.time()
    window = _rate_limit_store.get(requester, [])
    window = [t for t in window if now - t < 60.0]  # keep last 60s
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} queries/minute",
        )
    window.append(now)
    _rate_limit_store[requester] = window


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(
    org_id: Optional[str] = None,
    index: Optional[ZebraIndex] = None,
) -> FastAPI:
    """
    Create a FastAPI OrgShard application.

    Args:
        org_id: Identifier for this organization shard (e.g. 'OrgA').
                Falls back to ORG_ID environment variable.
        index:  Pre-built ZebraIndex. If None, an empty in-memory index is created.
    """
    _org_id = org_id or os.environ.get("ORG_ID", "UnknownOrg")
    _index  = index or ZebraIndex()
    _audit_db = _get_audit_db()

    app = FastAPI(
        title=f"ZebraID OrgShard — {_org_id}",
        description=(
            "Conservation organization shard for federated zebra re-identification. "
            "Accepts Z-Hash codes only; never returns raw images or GPS data."
        ),
        version="0.1.0",
    )

    # ── Health check ─────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"org_id": _org_id, "gallery_size": len(_index), "status": "ok"}

    # ── Match endpoint ────────────────────────────────────────────────────────
    @app.post("/match", response_model=MatchResponse)
    async def match(
        req: MatchRequest,
        api_key: str = Depends(_verify_api_key),
    ):
        _check_rate_limit(req.requester_org_id)

        # Decode hex Z-Hash — this is the ONLY data that crosses org boundaries
        try:
            z_hash_bytes = bytes.fromhex(req.z_hash_hex)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid hex Z-Hash")

        # Search the local index
        result = _index.search(z_hash_bytes)

        # Quantize scores → never return raw floats
        top_buckets = [_quantize_score(c["score"]) for c in result.top_k]
        primary_bucket = top_buckets[0] if top_buckets else SCORE_BUCKET_NO_MATCH
        primary_record_id = (
            result.top_k[0]["record_id"]
            if result.top_k and primary_bucket != SCORE_BUCKET_NO_MATCH
            else None
        )

        # Generate audit query ID
        query_id = hashlib.sha256(
            f"{_org_id}{req.requester_org_id}{req.z_hash_hex}{time.time()}".encode()
        ).hexdigest()[:16]

        # ── Audit log entry ───────────────────────────────────────────────────
        # Explicitly record that NO raw image and NO GPS was transmitted.
        _audit_db.execute(
            """INSERT INTO query_log VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                query_id,
                datetime.now(timezone.utc).isoformat(),
                req.requester_org_id,
                _org_id,
                len(z_hash_bytes),          # only bytes count is logged
                primary_bucket,
                primary_record_id,
                0,                           # raw_image_sent = FALSE
                0,                           # gps_sent = FALSE
            ),
        )
        _audit_db.commit()

        return MatchResponse(
            match_bucket=primary_bucket,
            record_id=primary_record_id,
            top_k_buckets=top_buckets,
            query_id=query_id,
        )

    # ── Audit log endpoint ───────────────────────────────────────────────────
    @app.get("/audit_log")
    async def audit_log(
        limit: int = 50,
        api_key: str = Depends(_verify_api_key),
    ):
        rows = _audit_db.execute(
            "SELECT * FROM query_log ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = [
            "query_id", "timestamp_utc", "requester_org", "this_org",
            "z_hash_bytes", "response_bucket", "record_id_returned",
            "raw_image_sent", "gps_sent",
        ]
        return {"org_id": _org_id, "entries": [dict(zip(cols, r)) for r in rows]}

    return app
