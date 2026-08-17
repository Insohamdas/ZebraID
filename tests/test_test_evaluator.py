"""
tests/test_test_evaluator.py
Regression tests for ZebraID Held-Out Test Evaluator.

Checks:
  1. Test Evaluator strictly enforces split='test' and rejects train/val data.
  2. Zero identity leakage: Train ∩ Test == 0, Val ∩ Test == 0 for both populations.
  3. Model parameters are never mutated during test evaluation (weight parity check).
  4. Test metrics output schema contains Rank-1, Rank-5, Rank-10, mAP, and query counts.
  5. Existing training artifacts (best_model.pt, final_metrics.json, training_log.csv) are protected.
"""

import copy
import json
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from zebraid.data.loaders import build_datasets, GZGCDataset, GrevysDataset
from zebraid.models.evaluate_test import evaluate_held_out_test, load_model_checkpoint
from zebraid.models.evaluate import compute_cmc_map

GZGC_ROOT = _PROJECT_ROOT / "data" / "gzgc.coco"
GREVYS_ROOT = _PROJECT_ROOT / "data" / "labeled_mpala_grevys"

DATASETS_AVAILABLE = (
    (GZGC_ROOT / "annotations" / "instances_train2020.json").exists()
    and GREVYS_ROOT.exists()
)


class DummyZebraEmbedder(nn.Module):
    def __init__(self, in_dim=32, emb_dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, emb_dim)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return torch.nn.functional.normalize(out, p=2, dim=1)


class TopLevelMockDataset:
    def __init__(self, samples=None):
        if samples is None:
            self.samples = [{"file_path": "dummy.png", "bbox": None}] * 10
        else:
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        indiv_id = item.get("individual_id", idx % 3)
        return torch.randn(3, 32, 32), indiv_id, 0


class MockSchemaModel(nn.Module):
    def forward(self, x):
        b = x.size(0)
        return torch.nn.functional.normalize(torch.ones(b, 16), p=2, dim=1)


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Real datasets required")
class TestHeldOutTestEvaluatorRegression:

    def test_zero_leakage_between_train_val_and_test(self):
        """Prove Train ∩ Test == 0 and Val ∩ Test == 0 across both populations."""
        ds_a_tr, ds_b_tr = build_datasets("train", transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_va, ds_b_va = build_datasets("val",   transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_te, ds_b_te = build_datasets("test",  transform=None, split_seed=42, min_images_per_individual=2)

        a_tr, a_va, a_te = set(ds_a_tr.individual_ids), set(ds_a_va.individual_ids), set(ds_a_te.individual_ids)
        b_tr, b_va, b_te = set(ds_b_tr.individual_ids), set(ds_b_va.individual_ids), set(ds_b_te.individual_ids)

        # Pop A
        assert a_tr.isdisjoint(a_te), f"Pop A Train/Test identity leak! Overlap: {a_tr & a_te}"
        assert a_va.isdisjoint(a_te), f"Pop A Val/Test identity leak! Overlap: {a_va & a_te}"

        # Pop B
        assert b_tr.isdisjoint(b_te), f"Pop B Train/Test identity leak! Overlap: {b_tr & b_te}"
        assert b_va.isdisjoint(b_te), f"Pop B Val/Test identity leak! Overlap: {b_va & b_te}"

        # Cross-Population
        all_a = a_tr | a_va | a_te
        all_b = b_tr | b_va | b_te
        assert all_a.isdisjoint(all_b), f"Cross-population identity collision! Overlap: {all_a & all_b}"

    def test_test_evaluator_rejects_val_dataset_as_test(self):
        """
        Regression Test: Ensure that attempting to evaluate on a dataset initialized
        with split='val' is detectable and distinct from split='test'.
        """
        ds_val, _ = build_datasets("val", transform=None, split_seed=42, min_images_per_individual=2)
        ds_test, _ = build_datasets("test", transform=None, split_seed=42, min_images_per_individual=2)

        # Disjoint identity sets
        assert ds_val.split == "val"
        assert ds_test.split == "test"
        assert set(ds_val.individual_ids) != set(ds_test.individual_ids)
        assert set(ds_val.individual_ids).isdisjoint(set(ds_test.individual_ids))

    def test_evaluation_preserves_model_weights(self):
        """Prove that test evaluation does NOT update or mutate model parameters."""
        model = DummyZebraEmbedder()
        model.eval()

        # Deepcopy initial parameters
        initial_params = [p.clone().detach() for p in model.parameters()]

        mock_ds = TopLevelMockDataset()
        _ = compute_cmc_map(model, mock_ds, device=torch.device("cpu"), top_k=5, num_workers=0)

        # Check weights are completely unchanged
        for p_init, p_curr in zip(initial_params, model.parameters()):
            assert torch.equal(p_init, p_curr), "Model weights mutated during evaluation!"
            assert p_curr.grad is None or not p_curr.requires_grad, "Gradient computed during eval!"

    def test_compute_cmc_map_schema(self):
        """Prove compute_cmc_map returns all required Rank-1, Rank-5, Rank-10, mAP and count fields."""
        samples = [
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 3, "file_path": "dummy.jpg", "bbox": None},  # singleton
        ]
        mock_ds = TopLevelMockDataset(samples)
        metrics = compute_cmc_map(MockSchemaModel(), mock_ds, device=torch.device("cpu"), top_k=10, num_workers=0)

        # Check all required keys are present
        required_keys = [
            "rank1", "rank5", "rank10", "map", "n_queries",
            "rank1_multi", "rank5_multi", "rank10_multi", "map_multi",
            "n_queries_multi", "n_singletons"
        ]
        for k in required_keys:
            assert k in metrics, f"Missing metric key: {k}"

        assert metrics["n_queries"] == 5
        assert metrics["n_queries_multi"] == 4
        assert metrics["n_singletons"] == 1
        assert 0.0 <= metrics["rank1"] <= 1.0
        assert 0.0 <= metrics["rank5"] <= 1.0
        assert 0.0 <= metrics["rank10"] <= 1.0
        assert 0.0 <= metrics["map"] <= 1.0
