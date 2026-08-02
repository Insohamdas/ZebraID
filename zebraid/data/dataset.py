"""
zebraid/data/dataset.py
ZebraDataset — unified COCO-format dataset loader for zebra re-identification.

Handles:
  - GZGC (population A) and a second population dataset (population B).
  - Stratified individual-level train / val / test splits (no individual
    appears in more than one split).
  - Returns (image_tensor, individual_id: int, population_label: int) tuples.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Label constants ──────────────────────────────────────────────────────────
POP_A = 0  # e.g., GZGC (plains zebra)
POP_B = 1  # e.g., Grevy's / mountain zebra


class ZebraDataset(Dataset):
    """
    Loads a single COCO-format zebra re-id dataset.

    Args:
        root:          Path to the dataset root (contains images/ and annotations/).
        annotation_file: Path to the COCO-format JSON annotation file.
        population_label: Integer label for this population (POP_A or POP_B).
        split:         One of 'train', 'val', 'test'.
        split_seed:    Random seed for reproducible splits.
        train_ratio:   Fraction of individuals used for training.
        val_ratio:     Fraction of individuals used for validation.
        transform:     Optional torchvision-compatible transform applied to images.
        individual_id_offset: Added to all individual IDs — use to ensure unique
                              IDs across two datasets when merging.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        population_label: int,
        split: Literal["train", "val", "test"] = "train",
        split_seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        transform: Optional[Callable] = None,
        individual_id_offset: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.population_label = population_label
        self.split = split
        self.transform = transform
        self.individual_id_offset = individual_id_offset

        # ── Load COCO annotations ────────────────────────────────────────────
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # Build image_id → file_name mapping
        id_to_filename: dict[int, str] = {
            img["id"]: img["file_name"] for img in coco["images"]
        }

        # Parse individual IDs from the "name" field in categories,
        # or from annotation "category_id" (depends on dataset convention).
        # We treat each category_id as one individual.
        category_id_to_name: dict[int, str] = {
            cat["id"]: cat.get("name", str(cat["id"]))
            for cat in coco.get("categories", [])
        }

        # Group annotations by individual (category_id)
        individual_to_samples: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            individual_to_samples[ann["category_id"]].append(
                {
                    "image_id": ann["image_id"],
                    "file_name": id_to_filename[ann["image_id"]],
                    "bbox": ann.get("bbox"),  # [x, y, w, h] or None
                }
            )

        # ── Stratified individual-level split ────────────────────────────────
        all_individuals = sorted(individual_to_samples.keys())
        rng = random.Random(split_seed)
        rng.shuffle(all_individuals)

        n = len(all_individuals)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_map = {
            "train": all_individuals[:n_train],
            "val": all_individuals[n_train : n_train + n_val],
            "test": all_individuals[n_train + n_val :],
        }
        selected_individuals = split_map[split]

        # ── Assign globally-stable integer IDs ──────────────────────────────────
        # Use position in the FULL sorted individual list so IDs are
        # non-overlapping across train/val/test splits.
        self._local_to_global: dict[int, int] = {
            local_id: (idx + individual_id_offset)
            for idx, local_id in enumerate(all_individuals)
        }

        self.samples: list[dict] = []
        for local_id in selected_individuals:
            global_id = self._local_to_global[local_id]
            for sample in individual_to_samples[local_id]:
                self.samples.append(
                    {
                        "file_path": self.root / "images" / sample["file_name"],
                        "individual_id": global_id,
                        "population_label": population_label,
                        "bbox": sample["bbox"],
                    }
                )

        self.num_individuals = len(selected_individuals)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        image = Image.open(sample["file_path"]).convert("RGB")

        # Crop to bounding box if available (tighter stripe region)
        if sample["bbox"] is not None:
            x, y, w, h = [int(v) for v in sample["bbox"]]
            image = image.crop((x, y, x + w, y + h))

        if self.transform is not None:
            image = self.transform(image)

        return image, sample["individual_id"], sample["population_label"]

    @property
    def individual_ids(self) -> list[int]:
        """Sorted list of unique individual IDs in this split."""
        return sorted({s["individual_id"] for s in self.samples})


class CombinedZebraDataset(Dataset):
    """
    Concatenates two ZebraDataset instances (population A + B) into one,
    ensuring individual IDs are globally unique across populations.

    Usage:
        ds_a = ZebraDataset(..., population_label=POP_A, individual_id_offset=0)
        ds_b = ZebraDataset(..., population_label=POP_B,
                            individual_id_offset=ds_a.num_individuals)
        combined = CombinedZebraDataset(ds_a, ds_b)
    """

    def __init__(self, dataset_a: ZebraDataset, dataset_b: ZebraDataset) -> None:
        self.datasets = [dataset_a, dataset_b]
        self._lengths = [len(dataset_a), len(dataset_b)]
        self._offsets = [0, len(dataset_a)]

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        for ds, offset in zip(self.datasets, self._offsets):
            if idx < offset + len(ds):
                return ds[idx - offset]
            # shouldn't reach here but keeps mypy happy
        raise IndexError(f"Index {idx} out of range for CombinedZebraDataset")
