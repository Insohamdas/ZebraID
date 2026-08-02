"""tests/test_loaders.py — Tests for real dataset loaders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Absolute paths resolved from the project root (parent of tests/)
_PROJECT_ROOT = Path(__file__).parent.parent
GZGC_ROOT     = _PROJECT_ROOT / "data" / "gzgc.coco"
GREVYS_ROOT   = _PROJECT_ROOT / "data" / "labeled_mpala_grevys"

# Skip all tests if datasets not present (CI environment)
pytestmark = pytest.mark.skipif(
    not (GZGC_ROOT / "annotations" / "instances_train2020.json").exists()
    or not GREVYS_ROOT.exists(),
    reason="Real datasets not available in this environment",
)

from zebraid.data.loaders import GZGCDataset, GrevysDataset, build_datasets
from zebraid.data.dataset import CombinedZebraDataset
from zebraid.data.mixed_batch_sampler import MixedPopulationBatchSampler
from zebraid.data.transforms import eval_transforms


class TestGZGCDataset:
    def test_sizes_reasonable(self):
        ds = GZGCDataset(root=GZGC_ROOT, split="train")
        assert ds.num_individuals >= 500, f"Expected ≥500 train individuals, got {ds.num_individuals}"
        assert len(ds) >= 2000, f"Expected ≥2000 train samples, got {len(ds)}"

    def test_category_filter_zebra_only(self):
        """Only plains zebra (category 1) annotations should be loaded."""
        ds = GZGCDataset(root=GZGC_ROOT, split="train")
        for s in ds.samples[:20]:
            assert s["population_label"] == 0

    def test_no_id_leakage_across_splits(self):
        train = GZGCDataset(root=GZGC_ROOT, split="train")
        val   = GZGCDataset(root=GZGC_ROOT, split="val")
        test  = GZGCDataset(root=GZGC_ROOT, split="test")
        assert set(train.individual_ids).isdisjoint(set(val.individual_ids))
        assert set(train.individual_ids).isdisjoint(set(test.individual_ids))
        assert set(val.individual_ids).isdisjoint(set(test.individual_ids))

    def test_image_tensor_shape(self):
        t = eval_transforms(img_size=224)
        ds = GZGCDataset(root=GZGC_ROOT, split="train", transform=t)
        img, ind_id, pop = ds[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(ind_id, int)
        assert pop == 0


class TestGrevysDataset:
    def test_sizes_reasonable(self):
        ds = GrevysDataset(root=GREVYS_ROOT, split="train")
        assert ds.num_individuals >= 100
        assert len(ds) >= 300

    def test_no_id_leakage_across_splits(self):
        train = GrevysDataset(root=GREVYS_ROOT, split="train")
        val   = GrevysDataset(root=GREVYS_ROOT, split="val")
        test  = GrevysDataset(root=GREVYS_ROOT, split="test")
        assert set(train.individual_ids).isdisjoint(set(val.individual_ids))
        assert set(train.individual_ids).isdisjoint(set(test.individual_ids))
        assert set(val.individual_ids).isdisjoint(set(test.individual_ids))

    def test_image_tensor_shape(self):
        t = eval_transforms(img_size=224)
        ds = GrevysDataset(root=GREVYS_ROOT, split="train", transform=t)
        img, ind_id, pop = ds[0]
        assert img.shape == (3, 224, 224)
        assert pop == 1   # POP_B


class TestBuildDatasets:
    def test_global_id_uniqueness_all_splits(self):
        """No ID collision across both populations AND across train/val/test."""
        seen = set()
        for split in ("train", "val", "test"):
            ds_a, ds_b = build_datasets(split, transform=None,
                                        gzgc_root=GZGC_ROOT, grevys_root=GREVYS_ROOT)
            ids = set(ds_a.individual_ids) | set(ds_b.individual_ids)
            overlap = seen & ids
            assert not overlap, f"Split '{split}' IDs overlap with previous splits: {overlap}"
            seen |= ids

    def test_grevys_ids_start_after_gzgc(self):
        """Grevy's IDs must be > max GZGC ID in any split."""
        for split in ("train", "val", "test"):
            ds_a, ds_b = build_datasets(split, transform=None,
                                        gzgc_root=GZGC_ROOT, grevys_root=GREVYS_ROOT)
            if ds_a.individual_ids and ds_b.individual_ids:
                assert min(ds_b.individual_ids) > max(ds_a.individual_ids), \
                    f"Grevy IDs overlap with GZGC in {split} split"

    def test_mixed_sampler_always_has_both_populations(self):
        t = eval_transforms(img_size=224)
        ds_a, ds_b = build_datasets("train", transform=t,
                                    gzgc_root=GZGC_ROOT, grevys_root=GREVYS_ROOT)
        combined = CombinedZebraDataset(ds_a, ds_b)
        sampler  = MixedPopulationBatchSampler(combined, batch_size=16)
        for i, batch_idx in enumerate(sampler):
            if i >= 5:
                break
            pops = {combined[j][2] for j in batch_idx}
            assert 0 in pops and 1 in pops, \
                f"Batch {i} missing a population: {pops}"
