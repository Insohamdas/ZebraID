"""tests/test_extra_features.py — Tests for the 4 extra features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from zebraid.data.detector import ZebraDetector, SpeciesClassifier
from zebraid.reporting.pdf_report import generate_sighting_report_html


def test_detector_returns_bounding_boxes():
    detector = ZebraDetector(confidence_threshold=0.3)
    # Synthetic test image
    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    detections = detector.detect_zebras(img)
    assert len(detections) >= 1
    assert "bbox" in detections[0]
    assert "crop" in detections[0]
    bbox = detections[0]["bbox"]
    assert len(bbox) == 4


def test_species_classifier():
    classifier = SpeciesClassifier()
    stripe_img = Image.new("RGB", (100, 100), color="white")
    result = classifier.classify(stripe_img)
    assert "species_name" in result
    assert result["population_label"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0


def test_sighting_report_generator():
    match_results = [{
        "individual_id": "IBEIS_PZ_1594",
        "match_confidence": "STRONG_MATCH",
        "species_name": "Plains Zebra",
        "raw_similarity_score": 0.94,
        "organization_shard": "Org A (Plains Zebra Shard)",
        "bbox": [120, 45, 680, 520],
    }]
    html = generate_sighting_report_html(match_results, query_id="QR-TEST-99")
    assert "ZebraID Sighting Report" in html
    assert "IBEIS_PZ_1594" in html
    assert "STRONG_MATCH" in html
    assert "QR-TEST-99" in html
    assert "Verifiable Privacy Audit Certificate" in html
