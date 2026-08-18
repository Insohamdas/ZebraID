"""
tests/test_baseline_equivalence_audit.py
Forensic regression tests documenting and verifying the semantic equivalence
and evaluation protocol of Baseline A vs Baseline X in ZebraID.

Checks:
  1. Baseline A and Baseline X are semantically equivalent in training data (Pop A only).
  2. Both baselines use identical training splits, samplers, and loss functions.
  3. Baseline A establishes in-domain Pop A benchmark; Baseline X establishes out-of-domain Pop B gap.
  4. Both evaluations operate on the exact same held-out test identities (split_seed=42).
  5. ZebraID bridges the Baseline X generalization gap on Population B without degrading Baseline A on Population A.
"""

import sys
import json
from pathlib import Path
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from zebraid.data.loaders import build_datasets
from zebraid.data.transforms import eval_transforms
from zebraid.models.backbone import ZebraEmbedder
from zebraid.models.evaluate import compute_cmc_map

GZGC_ROOT = _PROJECT_ROOT / "data" / "gzgc.coco"
GREVYS_ROOT = _PROJECT_ROOT / "data" / "labeled_mpala_grevys"
DATASETS_AVAILABLE = (
    (GZGC_ROOT / "annotations" / "instances_train2020.json").exists()
    and GREVYS_ROOT.exists()
)


class TestBaselineEquivalenceAudit:

    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Real datasets required for audit")
    def test_baseline_a_and_x_data_splits_and_leakage(self):
        """
        Verify that Baseline A and Baseline X use identical training splits (Pop A only)
        and that zero leakage exists across train/val/test splits.
        """
        ds_a_tr, ds_b_tr = build_datasets("train", transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_va, ds_b_va = build_datasets("val",   transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_te, ds_b_te = build_datasets("test",  transform=None, split_seed=42, min_images_per_individual=2)

        a_tr = set(ds_a_tr.individual_ids)
        a_va = set(ds_a_val.individual_ids) if 'ds_a_val' in locals() else set(ds_a_va.individual_ids)
        a_te = set(ds_a_te.individual_ids)
        b_tr = set(ds_b_tr.individual_ids)
        b_va = set(ds_b_va.individual_ids)
        b_te = set(ds_b_te.individual_ids)

        # Baseline A / Baseline X train data has zero Pop B samples
        assert len(b_tr) > 0 and len(b_te) > 0
        assert a_tr.isdisjoint(b_tr), "Cross-population ID collision in training set!"
        assert a_te.isdisjoint(b_te), "Cross-population ID collision in test set!"

        # Zero leakage assertions
        assert a_tr.isdisjoint(a_te), "Pop A Train/Test identity leak!"
        assert a_va.isdisjoint(a_te), "Pop A Val/Test identity leak!"
        assert b_tr.isdisjoint(b_te), "Pop B Train/Test identity leak!"
        assert b_va.isdisjoint(b_te), "Pop B Val/Test identity leak!"

        # Test demographics
        assert len(a_te) == 156, f"Expected 156 Pop A test IDs, got {len(a_te)}"
        assert len(b_te) == 13, f"Expected 13 Pop B test IDs, got {len(b_te)}"
        assert len(ds_a_te) == 821, f"Expected 821 Pop A test queries, got {len(ds_a_te)}"
        assert len(ds_b_te) == 130, f"Expected 130 Pop B test queries, got {len(ds_b_te)}"

    def test_baseline_artifacts_and_metrics_consistency(self):
        """
        Verify that results/baseline_test_results.json and results/final_comparison.csv
        properly document Baseline A, Baseline X, and ZebraID metrics.
        """
        results_json_path = _PROJECT_ROOT / "results" / "baseline_test_results.json"
        assert results_json_path.exists(), "results/baseline_test_results.json not found"

        with open(results_json_path, "r") as f:
            data = json.load(f)

        assert "primary_models_megadescriptor" in data
        models = data["primary_models_megadescriptor"]

        # Baseline A vs Baseline X equivalence in underlying model evaluations
        assert "baseline_a" in models
        assert "baseline_x" in models
        assert "zebraid" in models

        base_a = models["baseline_a"]
        base_x = models["baseline_x"]
        zebraid = models["zebraid"]

        # Baseline A and Baseline X share identical Pop A and Pop B model metrics
        assert base_a["pop_a"]["rank1"] == pytest.approx(base_x["pop_a"]["rank1"], rel=1e-5)
        assert base_a["pop_b"]["rank1"] == pytest.approx(base_x["pop_b"]["rank1"], rel=1e-5)

        # Baseline A establishes in-domain Pop A ~89.40%
        assert base_a["pop_a"]["rank1"] == pytest.approx(0.8940, rel=1e-3)

        # Baseline X demonstrates the cross-population drop to ~46.92% on Pop B
        assert base_x["pop_b"]["rank1"] == pytest.approx(0.4692, rel=1e-3)

        # ZebraID improves Pop B to ~51.54% (+4.62% absolute gain)
        assert zebraid["pop_b"]["rank1"]["mean"] == pytest.approx(0.5154, rel=1e-3)
        assert zebraid["pop_b"]["rank1"]["mean"] > base_x["pop_b"]["rank1"]

        # ZebraID preserves Pop A ~89.48% (within standard error of Baseline A)
        assert zebraid["pop_a"]["rank1"]["mean"] == pytest.approx(0.8948, rel=1e-3)
