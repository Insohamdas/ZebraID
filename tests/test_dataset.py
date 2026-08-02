"""tests/test_dataset.py — Component 7"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from zebraid.data.dataset import ZebraDataset, CombinedZebraDataset, POP_A, POP_B
from zebraid.data.mixed_batch_sampler import MixedPopulationBatchSampler


# Minimal valid 1×1 red PNG (parseable by PIL)
_VALID_PNG_BYTES = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
    b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)


def make_coco_fixture(tmp_path: Path, n_individuals: int = 20, images_per_ind: int = 4):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    images, anns, cats = [], [], []
    img_id, ann_id = 0, 0
    for ind in range(1, n_individuals + 1):
        cats.append({"id": ind, "name": f"zebra_{ind:03d}"})
        for _ in range(images_per_ind):
            img_id += 1
            fname = f"img_{img_id:04d}.png"
            (images_dir / fname).write_bytes(_VALID_PNG_BYTES)
            images.append({"id": img_id, "file_name": fname})
            anns.append({"id": ann_id, "image_id": img_id,
                         "category_id": ind, "bbox": [0, 0, 1, 1]})
            ann_id += 1

    ann_file = ann_dir / "instances.json"
    ann_file.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return tmp_path, ann_file


@pytest.fixture
def fixture_a(tmp_path):
    return make_coco_fixture(tmp_path / "popA", n_individuals=20)

@pytest.fixture
def fixture_b(tmp_path):
    return make_coco_fixture(tmp_path / "popB", n_individuals=15)


def test_no_individual_leaks_across_splits(fixture_a):
    root, ann = fixture_a
    ds_train = ZebraDataset(root, ann, POP_A, split="train")
    ds_val   = ZebraDataset(root, ann, POP_A, split="val")
    ds_test  = ZebraDataset(root, ann, POP_A, split="test")

    assert set(ds_train.individual_ids).isdisjoint(set(ds_val.individual_ids)), \
        "Leak: train → val"
    assert set(ds_train.individual_ids).isdisjoint(set(ds_test.individual_ids)), \
        "Leak: train → test"
    assert set(ds_val.individual_ids).isdisjoint(set(ds_test.individual_ids)), \
        "Leak: val → test"


def test_splits_cover_all_individuals(fixture_a):
    root, ann = fixture_a
    all_ids = set()
    for split in ("train", "val", "test"):
        all_ids |= set(ZebraDataset(root, ann, POP_A, split=split).individual_ids)
    assert len(all_ids) == 20, f"Not all individuals covered, got {len(all_ids)}"


def test_mixed_batch_sampler_always_has_both_populations(fixture_a, fixture_b):
    root_a, ann_a = fixture_a
    root_b, ann_b = fixture_b

    ds_a = ZebraDataset(root_a, ann_a, POP_A, "train")
    ds_b = ZebraDataset(root_b, ann_b, POP_B, "train",
                        individual_id_offset=ds_a.num_individuals)
    combined = CombinedZebraDataset(ds_a, ds_b)

    sampler = MixedPopulationBatchSampler(combined, batch_size=8, ratio_a=0.5)
    for batch_idx in sampler:
        pops = set()
        for i in batch_idx:
            # Read population label from dataset metadata (no image load)
            if i < len(ds_a):
                pops.add(ds_a.samples[i]["population_label"])
            else:
                pops.add(ds_b.samples[i - len(ds_a)]["population_label"])
        assert POP_A in pops, "Batch missing population A"
        assert POP_B in pops, "Batch missing population B"
