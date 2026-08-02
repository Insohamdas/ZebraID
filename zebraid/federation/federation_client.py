"""
zebraid/federation/federation_client.py
Cross-organization query client for the ZebraID federated matching protocol.

Sends a Z-Hash (bytes) to a remote OrgShard /match endpoint.
Returns a score bucket (NO_MATCH / POSSIBLE_MATCH / STRONG_MATCH) and
an opaque record_id — never a raw image or GPS coordinate.

This client is used by:
  - The demo coordinator (demo/app.py) to orchestrate cross-org queries.
  - The test suite (tests/test_federation.py) for correctness verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_API_KEY = os.environ.get("ZEBRAID_API_KEY", "demo-key-orga")


@dataclass
class CrossOrgMatchResult:
    """Result of a cross-organizational Z-Hash query."""
    target_org_url: str
    match_bucket: str                   # NO_MATCH | POSSIBLE_MATCH | STRONG_MATCH
    record_id: Optional[str]            # opaque, no GPS or individual name exposed
    top_k_buckets: list[str]
    query_id: str
    latency_ms: float
    bytes_transmitted: int              # Z-Hash bytes only — auditable
    success: bool
    error: Optional[str] = None


class FederationClient:
    """
    Sends Z-Hash queries to remote OrgShard instances.

    Privacy guarantees (enforced by design):
      - Only bytes.fromhex(z_hash) is transmitted — no raw image, no GPS.
      - bytes_transmitted is logged for every query.
      - Score buckets are returned (not raw float scores).

    Args:
        requester_org_id: This organization's identifier (for audit logs).
        api_key:          Shared API key for inter-org authentication.
        timeout:          HTTP request timeout in seconds.
        log_path:         Optional path to append a JSONL audit log locally.
    """

    def __init__(
        self,
        requester_org_id: str = "OrgA",
        api_key: str = DEFAULT_API_KEY,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        log_path: Optional[str] = None,
    ) -> None:
        self.requester_org_id = requester_org_id
        self._headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self, target_org_url: str, z_hash: bytes
    ) -> CrossOrgMatchResult:
        """
        Send a Z-Hash to a remote OrgShard /match endpoint.

        Args:
            target_org_url: Base URL of the target OrgShard (e.g. 'http://localhost:8002').
            z_hash:         Binary Z-Hash (bytes).
        Returns:
            CrossOrgMatchResult with match bucket, record_id, and audit info.
        """
        z_hash_hex = z_hash.hex()
        payload = {
            "z_hash_hex": z_hash_hex,
            "requester_org_id": self.requester_org_id,
        }
        bytes_transmitted = len(z_hash)  # measure BEFORE sending — auditable

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{target_org_url}/match",
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.perf_counter() - t0) * 1000
            result = CrossOrgMatchResult(
                target_org_url=target_org_url,
                match_bucket=data["match_bucket"],
                record_id=data.get("record_id"),
                top_k_buckets=data.get("top_k_buckets", []),
                query_id=data["query_id"],
                latency_ms=round(latency_ms, 3),
                bytes_transmitted=bytes_transmitted,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            result = CrossOrgMatchResult(
                target_org_url=target_org_url,
                match_bucket="ERROR",
                record_id=None,
                top_k_buckets=[],
                query_id="",
                latency_ms=round(latency_ms, 3),
                bytes_transmitted=bytes_transmitted,
                success=False,
                error=str(e),
            )
            logger.warning(f"[FederationClient] Query to {target_org_url} failed: {e}")

        self._log(result)
        return result

    async def query_async(
        self, target_org_url: str, z_hash: bytes
    ) -> CrossOrgMatchResult:
        """Async version for use in FastAPI coordinator."""
        z_hash_hex = z_hash.hex()
        payload = {
            "z_hash_hex": z_hash_hex,
            "requester_org_id": self.requester_org_id,
        }
        bytes_transmitted = len(z_hash)

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{target_org_url}/match",
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.perf_counter() - t0) * 1000
            result = CrossOrgMatchResult(
                target_org_url=target_org_url,
                match_bucket=data["match_bucket"],
                record_id=data.get("record_id"),
                top_k_buckets=data.get("top_k_buckets", []),
                query_id=data["query_id"],
                latency_ms=round(latency_ms, 3),
                bytes_transmitted=bytes_transmitted,
                success=True,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            result = CrossOrgMatchResult(
                target_org_url=target_org_url,
                match_bucket="ERROR",
                record_id=None,
                top_k_buckets=[],
                query_id="",
                latency_ms=round(latency_ms, 3),
                bytes_transmitted=bytes_transmitted,
                success=False,
                error=str(e),
            )

        self._log(result)
        return result

    # ── Audit logging ─────────────────────────────────────────────────────────

    def _log(self, result: CrossOrgMatchResult) -> None:
        """
        Log the cross-org query to a local JSONL file.
        Explicitly records: bytes_transmitted (Z-Hash only),
        raw_image_transmitted=false, gps_transmitted=false.
        """
        entry = {
            "timestamp": time.time(),
            "requester_org": self.requester_org_id,
            "target_org_url": result.target_org_url,
            "z_hash_bytes_transmitted": result.bytes_transmitted,
            "raw_image_transmitted": False,   # NEVER
            "gps_transmitted": False,          # NEVER
            "match_bucket": result.match_bucket,
            "record_id": result.record_id,
            "query_id": result.query_id,
            "latency_ms": result.latency_ms,
            "success": result.success,
            "error": result.error,
        }
        logger.info(f"[FederationClient] {entry}")

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
