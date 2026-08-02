"""tests/test_advanced_features.py — Tests for the 2 ultra-advanced features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from zebraid.visualization.heatmaps import StripeHeatmapGenerator
from zebraid.visualization.geo_map import GeoMigrationTracker


def test_stripe_heatmap_generator():
    gen = StripeHeatmapGenerator()
    img = Image.new("RGB", (128, 128), color="white")
    heatmap_b64 = gen.generate_heatmap_overlay(img)
    assert heatmap_b64.startswith("data:image/png;base64,")
    assert len(heatmap_b64) > 100


def test_geo_migration_tracker():
    tracker = GeoMigrationTracker()
    data = tracker.generate_migration_analytics(individual_id="INDIV_TEST_01", species_name="Plains Zebra")
    assert data["individual_id"] == "INDIV_TEST_01"
    assert "cross_park_migrations" in data
    assert len(data["nodes"]) >= 2
    assert "Privacy Verification" in data["privacy_status"] or "GPS" in data["privacy_status"]
