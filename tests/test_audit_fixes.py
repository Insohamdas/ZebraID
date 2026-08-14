"""
tests/test_audit_fixes.py
Comprehensive regression tests for ZebraID audit infrastructure fixes:
1. MixedPopulationBatchSampler epoch diversity and reproducibility
2. Pop-B singleton handling without fake duplicate positives
3. Evaluation safety with loud NaN/Inf rejection
4. Dual evaluation metrics (All Queries vs Multi-Image Queries)
5. Rich checkpoint saving, loading, and backward compatibility
"""

import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from zebraid.data.loaders import build_datasets, GZGCDataset, GrevysDataset
from zebraid.data.dataset import CombinedZebraDataset, ZebraDataset, POP_A, POP_B
from zebraid.data.mixed_batch_sampler import MixedPopulationBatchSampler
from zebraid.models.evaluate import compute_cmc_map
from zebraid.models.backbone import ZebraEmbedder

GZGC_ROOT = _PROJECT_ROOT / "data" / "gzgc.coco"
GREVYS_ROOT = _PROJECT_ROOT / "data" / "labeled_mpala_grevys"

DATASETS_AVAILABLE = (
    (GZGC_ROOT / "annotations" / "instances_train2020.json").exists()
    and GREVYS_ROOT.exists()
)


# ── 1. Sampler Epoch Diversity & Reproducibility ─────────────────────────────

@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Real datasets required")
class TestMixedPopulationBatchSamplerFixes:
    def setup_method(self):
        ds_a, ds_b = build_datasets(
            "train", transform=None, split_seed=42, min_images_per_individual=2,
            gzgc_root=GZGC_ROOT, grevys_root=GREVYS_ROOT
        )
        self.combined = CombinedZebraDataset(ds_a, ds_b)
        self.sampler = MixedPopulationBatchSampler(
            self.combined, batch_size=16, ratio_a=0.5, seed=42
        )

    def test_sampler_epoch_diversity(self):
        """Prove batch_sequence(epoch=0) != batch_sequence(epoch=1)."""
        self.sampler.set_epoch(0)
        batches_epoch_0 = [batch.copy() for batch in self.sampler]

        self.sampler.set_epoch(1)
        batches_epoch_1 = [batch.copy() for batch in self.sampler]

        assert batches_epoch_0 != batches_epoch_1, (
            "CRITICAL: Epoch 0 and Epoch 1 produced identical batch sequences! "
            "Sampler must generate diverse batches across epochs."
        )

    def test_sampler_reproducibility(self):
        """Prove batch_sequence(seed=42, epoch=0) == batch_sequence(seed=42, epoch=0)."""
        sampler1 = MixedPopulationBatchSampler(self.combined, batch_size=16, ratio_a=0.5, seed=42)
        sampler1.set_epoch(0)
        seq1 = [b.copy() for b in sampler1]

        sampler2 = MixedPopulationBatchSampler(self.combined, batch_size=16, ratio_a=0.5, seed=42)
        sampler2.set_epoch(0)
        seq2 = [b.copy() for b in sampler2]

        assert seq1 == seq2, "Determinism failed: Same seed and epoch must yield identical batches."

    def test_no_singleton_fake_positives(self):
        """Prove no identity inside any batch is an artificially duplicated singleton."""
        self.sampler.set_epoch(0)
        # Build global index to sample metadata lookup without opening image files
        flat_samples = []
        for ds in self.combined.datasets:
            flat_samples.extend(ds.samples)

        for batch_idx, batch in enumerate(self.sampler):
            id_to_indices = {}
            for sample_idx in batch:
                sample = flat_samples[sample_idx]
                indiv_id = sample["individual_id"]
                if indiv_id not in id_to_indices:
                    id_to_indices[indiv_id] = []
                id_to_indices[indiv_id].append(sample_idx)

            for indiv_id, indices in id_to_indices.items():
                assert len(indices) == len(set(indices)), (
                    f"Batch {batch_idx} contains duplicated image index for individual {indiv_id}: "
                    f"indices={indices}. Singletons must not be duplicated into fake positive pairs."
                )


# ── 2. Pop-B Singleton Handling & Loader Stats ────────────────────────────────

@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Real datasets required")
class TestSingletonHandling:
    def test_grevys_loader_stats(self):
        """Verify GrevysDataset reports total, eligible, and excluded singletons."""
        ds = GrevysDataset(
            root=GREVYS_ROOT, split="train", min_images_per_individual=2, split_seed=42
        )
        assert hasattr(ds, "total_individuals")
        assert hasattr(ds, "eligible_individuals")
        assert hasattr(ds, "excluded_singletons")
        assert ds.total_individuals == ds.eligible_individuals + ds.excluded_singletons
        assert ds.excluded_singletons > 0, "Expected excluded singletons when min_images=2"
        assert ds.total_individuals == 173
        assert ds.eligible_individuals == 77
        assert ds.excluded_singletons == 96
        assert ds.num_individuals == 53  # 70% train split of 77 eligible individuals

        stats = ds.stats()
        assert stats["total_individuals"] == 173
        assert stats["eligible_individuals"] == 77
        assert stats["excluded_singletons"] == 96
        assert stats["n_individuals"] == 53


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Real datasets required")
class TestDatasetSplitLeakage:
    """Dedicated regression tests ensuring zero train/val/test identity leakage."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_zero_identity_leakage_across_all_splits(self, seed: int):
        """Prove Train ∩ Val == 0, Train ∩ Test == 0, Val ∩ Test == 0 for Pop A & Pop B."""
        ds_a_train, ds_b_train = build_datasets("train", transform=None, split_seed=seed, min_images_per_individual=2)
        ds_a_val,   ds_b_val   = build_datasets("val",   transform=None, split_seed=seed, min_images_per_individual=2)
        ds_a_test,  ds_b_test  = build_datasets("test",  transform=None, split_seed=seed, min_images_per_individual=2)

        # Pop A Disjointness
        a_tr = set(ds_a_train.individual_ids)
        a_va = set(ds_a_val.individual_ids)
        a_te = set(ds_a_test.individual_ids)

        assert a_tr.isdisjoint(a_va), f"Seed {seed}: Pop A Train/Val identity leak! Overlap: {a_tr & a_va}"
        assert a_tr.isdisjoint(a_te), f"Seed {seed}: Pop A Train/Test identity leak! Overlap: {a_tr & a_te}"
        assert a_va.isdisjoint(a_te), f"Seed {seed}: Pop A Val/Test identity leak! Overlap: {a_va & a_te}"

        # Pop B Disjointness
        b_tr = set(ds_b_train.individual_ids)
        b_va = set(ds_b_val.individual_ids)
        b_te = set(ds_b_test.individual_ids)

        assert b_tr.isdisjoint(b_va), f"Seed {seed}: Pop B Train/Val identity leak! Overlap: {b_tr & b_va}"
        assert b_tr.isdisjoint(b_te), f"Seed {seed}: Pop B Train/Test identity leak! Overlap: {b_tr & b_te}"
        assert b_va.isdisjoint(b_te), f"Seed {seed}: Pop B Val/Test identity leak! Overlap: {b_va & b_te}"

        # Cross-Population Global ID Disjointness
        all_a = a_tr | a_va | a_te
        all_b = b_tr | b_va | b_te
        assert all_a.isdisjoint(all_b), f"Seed {seed}: Global Pop A and Pop B ID collision! Overlap: {all_a & all_b}"

    def test_exact_measured_split_counts(self):
        """Verify exact measured image and identity counts on default seed 42."""
        ds_a_tr, ds_b_tr = build_datasets("train", transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_va, ds_b_va = build_datasets("val",   transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_te, ds_b_te = build_datasets("test",  transform=None, split_seed=42, min_images_per_individual=2)

        # Pop A
        assert len(ds_a_tr) == 3796
        assert ds_a_tr.num_individuals == 723
        assert len(ds_a_va) == 797
        assert ds_a_va.num_individuals == 154
        assert len(ds_a_te) == 821
        assert ds_a_te.num_individuals == 156

        # Pop B
        assert len(ds_b_tr) == 369
        assert ds_b_tr.num_individuals == 53
        assert len(ds_b_va) == 90
        assert ds_b_va.num_individuals == 11
        assert len(ds_b_te) == 130
        assert ds_b_te.num_individuals == 13


# ── 3. Evaluation Safety & Loud NaN/Inf Rejection ─────────────────────────────

class MockDataset:
    """Lightweight mock dataset for fast unit testing of evaluate.py."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return (dummy_image_tensor, individual_id, pop_label)
        return torch.zeros(3, 32, 32), self.samples[idx]["individual_id"], 0


class MockModelWithNaN(nn.Module):
    def __init__(self, inject_nan=True):
        super().__init__()
        self.inject_nan = inject_nan

    def forward(self, x):
        b = x.size(0)
        out = torch.ones(b, 16)
        if self.inject_nan:
            out[0, 0] = float("nan")
        return out


class TestEvaluationSafety:
    def test_nan_embedding_rejection(self):
        """Prove compute_cmc_map raises RuntimeError immediately if embeddings contain NaN."""
        dataset = MockDataset([
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None},
        ])
        model = MockModelWithNaN(inject_nan=True)

        with pytest.raises(RuntimeError, match="Evaluation Safety Check Failed"):
            compute_cmc_map(model, dataset, device=torch.device("cpu"))

    def test_clean_embedding_passes(self):
        """Prove compute_cmc_map succeeds when embeddings are finite."""
        dataset = MockDataset([
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None},
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None},
        ])
        model = MockModelWithNaN(inject_nan=False)

        metrics = compute_cmc_map(model, dataset, device=torch.device("cpu"), top_k=2)
        assert "rank1" in metrics
        assert "rank1_multi" in metrics
        assert "map" in metrics
        assert "map_multi" in metrics
        assert np.isfinite(metrics["rank1"])


# ── 4. Dual Evaluation Reporting (All Queries vs Multi-Image) ─────────────────

class DeterministicMockModel(nn.Module):
    def __init__(self, embs):
        super().__init__()
        self.embs = torch.tensor(embs, dtype=torch.float32)
        self.call_count = 0

    def forward(self, x):
        # Return pre-defined normalized embeddings
        b = x.size(0)
        out = self.embs[self.call_count : self.call_count + b]
        self.call_count += b
        return torch.nn.functional.normalize(out, p=2, dim=1)


class TestDualEvaluationReporting:
    def test_all_vs_multi_image_metrics(self):
        """
        Verify that singletons are included in 'All Queries' (and fail due to 0 matches)
        while being excluded from 'Multi-Image Queries Only'.
        """
        # 3 samples:
        # ID 1 has 2 images (multi-image): sample 0 and sample 1 (embeddings identical -> Rank-1 match)
        # ID 2 has 1 image (singleton): sample 2 (no match possible)
        dataset = MockDataset([
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None}, # query 0
            {"individual_id": 1, "file_path": "dummy.jpg", "bbox": None}, # query 1
            {"individual_id": 2, "file_path": "dummy.jpg", "bbox": None}, # query 2 (singleton)
        ])

        # Feature vectors in 2D:
        # Sample 0: [1.0, 0.0]
        # Sample 1: [1.0, 0.0]
        # Sample 2: [0.0, 1.0]
        embs = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        model = DeterministicMockModel(embs)

        metrics = compute_cmc_map(model, dataset, device=torch.device("cpu"), batch_size=4)

        # All queries: 2 correct out of 3 -> Rank-1 = 2/3 = 66.67%
        assert pytest.approx(metrics["rank1"], 0.01) == 2 / 3
        assert metrics["n_queries"] == 3

        # Multi-image queries only: 2 correct out of 2 -> Rank-1_multi = 100.0%
        assert pytest.approx(metrics["rank1_multi"], 0.01) == 1.0
        assert metrics["n_queries_multi"] == 2
        assert metrics["n_singletons"] == 1


# ── 5. Checkpoint Saving & Resumption ────────────────────────────────────────

class TestCheckpointManagement:
    def test_rich_checkpoint_save_and_load(self, tmp_path):
        """Prove rich checkpoint payload stores model, optimizer, scheduler, epoch, and config."""
        model = nn.Sequential(nn.Linear(10, 5))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        ckpt_path = tmp_path / "best_model.pt"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 5,
            "best_metrics": {"rank1_a": 0.92, "rank1_b": 0.55},
            "rng_state": torch.get_rng_state(),
            "config": {"model": "megadescriptor"},
        }
        torch.save(payload, ckpt_path)

        # Verify load
        loaded = torch.load(ckpt_path, map_location="cpu")
        assert "model" in loaded
        assert "optimizer" in loaded
        assert "scheduler" in loaded
        assert loaded["epoch"] == 5
        assert loaded["best_metrics"]["rank1_a"] == 0.92

        # Verify new model can load state dict
        new_model = nn.Sequential(nn.Linear(10, 5))
        if isinstance(loaded, dict) and "model" in loaded:
            new_model.load_state_dict(loaded["model"])
        else:
            new_model.load_state_dict(loaded)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_legacy_state_dict_backward_compatibility(self, tmp_path):
        """Prove backward compatibility with legacy state_dict-only checkpoints."""
        model = nn.Sequential(nn.Linear(10, 5))
        ckpt_path = tmp_path / "legacy_model.pt"
        torch.save(model.state_dict(), ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu")
        new_model = nn.Sequential(nn.Linear(10, 5))
        if isinstance(loaded, dict) and "model" in loaded:
            new_model.load_state_dict(loaded["model"])
        else:
            new_model.load_state_dict(loaded)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
