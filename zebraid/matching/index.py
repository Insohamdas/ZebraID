"""
zebraid/matching/index.py
ZebraIndex — FAISS-backed matching engine for Z-Hash codes.

Supports:
  - Exact matching (IndexFlatL2) for real dataset scale.
  - Approximate matching (IndexIVFPQ / HNSW) for scale simulation benchmarks.
  - Human-in-the-loop confidence thresholding: below the configured similarity
    threshold, returns NEEDS_REVIEW with a ranked candidate list rather than
    auto-confirming a match.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MatchStatus(str, Enum):
    CONFIRMED = "CONFIRMED"          # similarity ≥ threshold → auto-confirm
    NEEDS_REVIEW = "NEEDS_REVIEW"    # similarity < threshold → human review
    NO_MATCH = "NO_MATCH"            # no candidates above floor threshold


@dataclass
class MatchResult:
    status: MatchStatus
    top_k: list[dict]               # [{"record_id": str, "score": float, "rank": int}]
    query_time_ms: float
    code_size_bytes: int


class ZebraIndex:
    """
    FAISS-backed index for Z-Hash matching.

    Args:
        embedding_dim:        Dimensionality of the full embedding (for reference).
        code_size_bytes:      Size of each Z-Hash code in bytes (e.g., 32 for 256b).
        index_type:           'flat_l2' | 'ivf_pq' | 'hnsw'.
        confidence_threshold: Minimum similarity score to auto-confirm a match.
                              Below this, result is tagged NEEDS_REVIEW.
        top_k:                Number of candidates to return.
        db_path:              Path to SQLite metadata database.
    """

    def __init__(
        self,
        code_size_bytes: int = 32,
        index_type: str = "flat_l2",
        confidence_threshold: float = 0.75,
        top_k: int = 5,
        db_path: str = ":memory:",
    ) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install faiss-cpu: pip install faiss-cpu")

        self.code_size_bytes = code_size_bytes
        self.index_type = index_type
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        # Binary codes are stored as byte arrays; FAISS binary index uses bits
        self.n_bits = code_size_bytes * 8

        # ── Build FAISS index ────────────────────────────────────────────────
        if index_type == "flat_l2":
            # Exact Hamming distance search over binary codes
            self._index = faiss.IndexBinaryFlat(self.n_bits)
        elif index_type == "ivf_pq":
            # Approximate: inverted file + product quantization
            # nlist = sqrt(N) rule of thumb; set to 100 for prototype
            quantizer = faiss.IndexBinaryFlat(self.n_bits)
            self._index = faiss.IndexBinaryIVF(quantizer, self.n_bits, nlist=100)
            self._index.nprobe = 10
        elif index_type == "hnsw":
            # HNSW does not have a binary variant in faiss-cpu;
            # use float32 index with binary codes unpacked to float
            self._use_hnsw_float = True
            self._index = faiss.IndexHNSWFlat(self.n_bits, 32)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        self._use_hnsw_float = index_type == "hnsw"
        self._n_indexed = 0

        # ── SQLite metadata store ─────────────────────────────────────────────
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS gallery (
                faiss_id    INTEGER PRIMARY KEY,
                record_id   TEXT NOT NULL,
                individual_id INTEGER,
                population  INTEGER,
                extra_json  TEXT
            )
        """)
        self._db.commit()

    # ── Adding records ───────────────────────────────────────────────────────

    def add(
        self,
        z_hash: bytes,
        record_id: str,
        individual_id: Optional[int] = None,
        population: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> int:
        """
        Add a Z-Hash to the index.

        Args:
            z_hash:        Binary Z-Hash (bytes), length must equal code_size_bytes.
            record_id:     Opaque string ID for the record (returned in match results).
            individual_id: Optional integer individual ID.
            population:    Optional integer population label (0=A, 1=B).
            extra:         Optional dict of additional metadata.
        Returns:
            FAISS ID assigned to this record.
        """
        if len(z_hash) != self.code_size_bytes:
            raise ValueError(
                f"z_hash must be {self.code_size_bytes} bytes, got {len(z_hash)}"
            )
        code = np.frombuffer(z_hash, dtype=np.uint8).reshape(1, -1)

        if self._use_hnsw_float:
            vec = np.unpackbits(code, axis=1).astype(np.float32)
            self._index.add(vec)
        else:
            self._index.add(code)

        faiss_id = self._n_indexed
        self._n_indexed += 1

        self._db.execute(
            "INSERT INTO gallery VALUES (?, ?, ?, ?, ?)",
            (faiss_id, record_id, individual_id, population,
             json.dumps(extra) if extra else None),
        )
        self._db.commit()
        return faiss_id

    def add_batch(
        self,
        z_hashes: np.ndarray,
        record_ids: list[str],
        individual_ids: Optional[list[int]] = None,
        populations: Optional[list[int]] = None,
    ) -> None:
        """Add a batch of Z-Hash codes."""
        assert z_hashes.shape[0] == len(record_ids)
        z_hashes = z_hashes.astype(np.uint8)

        if self._use_hnsw_float:
            vecs = np.unpackbits(z_hashes, axis=1).astype(np.float32)
            self._index.add(vecs)
        else:
            self._index.add(z_hashes)

        rows = []
        for i, rid in enumerate(record_ids):
            faiss_id = self._n_indexed + i
            rows.append((
                faiss_id, rid,
                individual_ids[i] if individual_ids else None,
                populations[i] if populations else None,
                None,
            ))
        self._db.executemany("INSERT INTO gallery VALUES (?, ?, ?, ?, ?)", rows)
        self._db.commit()
        self._n_indexed += len(record_ids)

    # ── Searching ────────────────────────────────────────────────────────────

    def search(self, query_z_hash: bytes) -> MatchResult:
        """
        Search for the closest matching Z-Hash in the index.

        Args:
            query_z_hash: Binary Z-Hash (bytes), same size as indexed codes.
        Returns:
            MatchResult with status, top-k candidates, and timing.
        """
        t0 = time.perf_counter()
        code = np.frombuffer(query_z_hash, dtype=np.uint8).reshape(1, -1)

        if self._use_hnsw_float:
            vec = np.unpackbits(code, axis=1).astype(np.float32)
            distances, faiss_ids = self._index.search(vec, self.top_k)
        else:
            distances, faiss_ids = self._index.search(code, self.top_k)

        query_time_ms = (time.perf_counter() - t0) * 1000

        # Convert Hamming distances to similarity scores [0, 1]
        max_hamming = float(self.n_bits)
        candidates = []
        for rank, (dist, fid) in enumerate(
            zip(distances[0].tolist(), faiss_ids[0].tolist())
        ):
            if fid < 0:
                continue  # FAISS returns -1 for padding
            similarity = 1.0 - (dist / max_hamming)
            row = self._db.execute(
                "SELECT record_id, individual_id, population FROM gallery WHERE faiss_id=?",
                (fid,),
            ).fetchone()
            if row:
                candidates.append({
                    "record_id": row[0],
                    "individual_id": row[1],
                    "population": row[2],
                    "score": round(similarity, 4),
                    "rank": rank + 1,
                })

        # ── Human-in-the-loop threshold ──────────────────────────────────────
        if not candidates:
            status = MatchStatus.NO_MATCH
        elif candidates[0]["score"] >= self.confidence_threshold:
            status = MatchStatus.CONFIRMED
        else:
            status = MatchStatus.NEEDS_REVIEW

        return MatchResult(
            status=status,
            top_k=candidates,
            query_time_ms=round(query_time_ms, 3),
            code_size_bytes=self.code_size_bytes,
        )

    def __len__(self) -> int:
        return self._n_indexed
