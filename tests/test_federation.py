"""tests/test_federation.py — Component 7 (unit-level, no network)"""
import sys, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zebraid.federation.org_service import create_app
from zebraid.matching.index import ZebraIndex, MatchStatus


VALID_KEY = "demo-key-orga"
HEADERS   = {"X-API-Key": VALID_KEY}


def _make_shard(n_records: int, code_size_bytes: int = 32, org_id: str = "TestOrg"):
    import os; os.environ["ZEBRAID_API_KEYS"] = VALID_KEY
    index = ZebraIndex(code_size_bytes=code_size_bytes, confidence_threshold=0.55)
    rng = np.random.default_rng(123)
    codes = []
    for i in range(n_records):
        code = bytes(rng.integers(0, 256, code_size_bytes, dtype=np.uint8))
        index.add(code, record_id=f"rec_{org_id}_{i}", individual_id=i, population=0)
        codes.append(code)
    app = create_app(org_id=org_id, index=index)
    return TestClient(app), codes, index


class TestCrossShardMatch:
    def test_known_individual_resolves_in_same_shard(self):
        client, codes, index = _make_shard(10)
        # Query with exact code of individual 3
        resp = client.post("/match", json={
            "z_hash_hex": codes[3].hex(),
            "requester_org_id": "OrgB",
        }, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        # With identical code, similarity = 1.0 → should be STRONG_MATCH
        assert data["match_bucket"] in ("STRONG_MATCH", "POSSIBLE_MATCH"), \
            f"Expected match for known individual, got {data['match_bucket']}"

    def test_unknown_individual_produces_no_strong_match(self):
        client, codes, _ = _make_shard(10)
        # Completely random code — unlikely to match
        rng = np.random.default_rng(9999)
        unknown_code = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        resp = client.post("/match", json={
            "z_hash_hex": unknown_code.hex(),
            "requester_org_id": "OrgB",
        }, headers=HEADERS)
        assert resp.status_code == 200
        # Should not be STRONG_MATCH (might be POSSIBLE or NO_MATCH due to random similarity)
        data = resp.json()
        assert data["match_bucket"] != "STRONG_MATCH"

    def test_audit_log_records_every_query(self):
        client, codes, _ = _make_shard(5)
        n_queries = 3
        for i in range(n_queries):
            client.post("/match", json={
                "z_hash_hex": codes[i].hex(),
                "requester_org_id": "AuditTestOrg",
            }, headers=HEADERS)

        audit = client.get("/audit_log", headers=HEADERS)
        assert audit.status_code == 200
        entries = audit.json()["entries"]
        assert len(entries) >= n_queries, f"Expected ≥{n_queries} audit entries, got {len(entries)}"

    def test_audit_log_records_zero_raw_image_and_gps(self):
        client, codes, _ = _make_shard(5)
        client.post("/match", json={
            "z_hash_hex": codes[0].hex(),
            "requester_org_id": "PrivacyTestOrg",
        }, headers=HEADERS)

        audit = client.get("/audit_log", headers=HEADERS)
        entries = audit.json()["entries"]
        for entry in entries:
            assert entry["raw_image_sent"] == 0, "Audit log shows raw image was transmitted!"
            assert entry["gps_sent"] == 0,       "Audit log shows GPS was transmitted!"

    def test_health_endpoint(self):
        client, _, _ = _make_shard(5)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["gallery_size"] == 5
