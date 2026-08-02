"""tests/test_demo_app.py — Unit tests for demo app API routes."""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from fastapi.testclient import TestClient
from demo.app import app

client = TestClient(app)


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "ZebraID" in response.text


def test_identify_route():
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    response = client.post("/identify", files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert len(data["detections"]) >= 1
    assert "individual_id" in data["detections"][0]
    assert "species_name" in data["detections"][0]
    assert "heatmap_b64" in data["detections"][0]


def test_report_route():
    payload = {
        "detections": [{
            "individual_id": "IBEIS_PZ_1594",
            "match_confidence": "STRONG_MATCH",
            "species_name": "Plains Zebra",
            "raw_similarity_score": 0.94,
            "organization_shard": "Org A (Plains Zebra Shard)",
            "bbox": [120, 45, 680, 520],
        }]
    }
    response = client.post("/api/report", json=payload)
    assert response.status_code == 200
    assert "ZebraID Sighting Report" in response.text
