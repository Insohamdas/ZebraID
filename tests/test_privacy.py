"""tests/test_privacy.py — Component 7"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
import numpy as np

from zebraid.federation.org_service import create_app
from zebraid.matching.index import ZebraIndex, MatchStatus


VALID_KEY = "demo-key-orga"
VALID_HEADERS = {"X-API-Key": VALID_KEY}


@pytest.fixture
def org_client():
    """Create a test OrgShard with a seeded index."""
    index = ZebraIndex(code_size_bytes=32, confidence_threshold=0.75)

    # Seed 5 records
    rng = np.random.default_rng(0)
    for i in range(5):
        code = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        index.add(code, record_id=f"record_{i}", individual_id=i, population=0)

    import os; os.environ["ZEBRAID_API_KEYS"] = VALID_KEY
    app = create_app(org_id="TestOrg", index=index)
    return TestClient(app)


def test_match_response_never_contains_raw_image(org_client):
    code = bytes(np.random.default_rng(99).integers(0, 256, 32, dtype=np.uint8))
    resp = org_client.post("/match", json={
        "z_hash_hex": code.hex(),
        "requester_org_id": "TestRequester",
    }, headers=VALID_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    # Verify no image-related fields in response
    response_str = json.dumps(data).lower()
    assert "image" not in response_str or "image" == "image" * 0, \
        "Response contains 'image' key — potential data leak"
    assert "pixel" not in response_str, "Response contains 'pixel' — potential image data"
    assert "base64" not in response_str, "Response contains base64 — potential image encoding"


def test_match_response_never_contains_gps(org_client):
    code = bytes(np.random.default_rng(77).integers(0, 256, 32, dtype=np.uint8))
    resp = org_client.post("/match", json={
        "z_hash_hex": code.hex(),
        "requester_org_id": "TestRequester",
    }, headers=VALID_HEADERS)
    data = resp.json()
    response_str = json.dumps(data).lower()

    assert "latitude" not in response_str
    assert "longitude" not in response_str
    assert "gps" not in response_str
    assert "location" not in response_str


def test_match_response_uses_buckets_not_raw_scores(org_client):
    code = bytes(np.random.default_rng(55).integers(0, 256, 32, dtype=np.uint8))
    resp = org_client.post("/match", json={
        "z_hash_hex": code.hex(),
        "requester_org_id": "TestRequester",
    }, headers=VALID_HEADERS)
    data = resp.json()

    valid_buckets = {"NO_MATCH", "POSSIBLE_MATCH", "STRONG_MATCH"}
    assert data["match_bucket"] in valid_buckets, \
        f"match_bucket '{data['match_bucket']}' is not a valid bucket"

    for b in data.get("top_k_buckets", []):
        assert b in valid_buckets, f"top_k_buckets contains raw bucket '{b}'"

    # Verify no float similarity score is returned directly
    assert "score" not in data, "Raw similarity score exposed in response"
    assert "similarity" not in data, "Raw similarity exposed in response"
    assert "distance" not in data, "Raw distance exposed in response"


def test_auth_required(org_client):
    code = bytes(32)
    resp = org_client.post("/match", json={
        "z_hash_hex": code.hex(),
        "requester_org_id": "attacker",
    })  # no API key header
    assert resp.status_code == 401


def test_invalid_api_key_rejected(org_client):
    code = bytes(32)
    resp = org_client.post("/match", json={
        "z_hash_hex": code.hex(),
        "requester_org_id": "attacker",
    }, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
