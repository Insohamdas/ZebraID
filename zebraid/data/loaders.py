"""
zebraid/data/loaders.py
Dataset-specific loaders for the two real datasets used in ZebraID.

Population A — GZGC (Great Zebra & Giraffe Count)
  Format: COCO JSON with per-annotation 'name' field as individual identity.
  Path:   data/gzgc.coco/

Population B — Labeled Mpala Grevy's Zebra
  Format: Folder-per-individual layout. No COCO file needed.
          labeled_mpala_grevys/<INDIV_ID>/<ANNOT_ID>.png
  Path:   labeled_mpala_grevys/

Both loaders produce the same output tuple:
  (image_tensor, individual_id: int, population_label: int)

Usage (for training):
    from zebraid.data.loaders import GZGCDataset, GrevysDataset
    from zebraid.data.dataset import CombinedZebraDataset

    ds_a_train = GZGCDataset(split='train', transform=train_transforms())
    ds_b_train = GrevysDataset(split='train', transform=train_transforms(),
                               individual_id_offset=ds_a_train.num_individuals)
    combined   = CombinedZebraDataset(ds_a_train, ds_b_train)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from zebraid.data.dataset import POP_A, POP_B

Split = Literal["train", "val", "test"]


# ─────────────────────────────────────────────────────────────────────────────
# Population A — GZGC
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_gzgc_root() -> Path:
    """Resolve GZGC dataset root at call-time (not import-time) so CWD is correct."""
    for p in [Path("data/gzgc.coco"), Path("gzgc.coco")]:
        if p.exists():
            return p
    return Path("data/gzgc.coco")

GZGC_CATEGORY_ID_ZEBRA = 1   # category_id 1 = zebra_plains


class GZGCDataset(Dataset):
    """
    GZGC (Great Zebra & Giraffe Count) dataset loader.

    Individual identity is determined by the 'name' field in each annotation
    (e.g. 'IBEIS_PZ_1594'). This is the standard IBEIS individual identifier
    used in Wildbook / Wild-ID literature — it is the correct identity signal
    for re-identification, not the annotation id.

    Only zebra_plains annotations (category_id == 1) are loaded.
    Giraffe annotations are ignored.

    Stats:
      ~1,905 unique individuals · ~6,286 zebra annotations · 4,948 images

    Args:
        root:       Path to the gzgc.coco directory. Defaults to 'data/gzgc.coco'.
        ann_file:   Path to COCO JSON annotation file. Defaults to
                    instances_train2020.json.
        image_dir:  Path to the folder containing JPG images.
        split:      'train' | 'val' | 'test'. Stratified by individual.
        split_seed: Random seed for reproducible splits.
        train_ratio / val_ratio: Fractions of individuals per split.
        transform:  torchvision-compatible transform applied to images.
        min_images_per_individual: Drop individuals with fewer images.
                    Default 2 (triplet loss needs at least 2 images per individual).
        individual_id_offset: Added to all individual IDs so they remain globally
                    unique when combined with another dataset.
    """

    def __init__(
        self,
        root:           Optional[str | Path] = None,
        ann_file:       Optional[str | Path] = None,
        image_dir:      Optional[str | Path] = None,
        split:          Split = "train",
        split_seed:     int = 42,
        train_ratio:    float = 0.70,
        val_ratio:      float = 0.15,
        transform:      Optional[Callable] = None,
        min_images_per_individual: int = 2,
        individual_id_offset: int = 0,
    ) -> None:
        super().__init__()
        # Resolve root lazily so os.chdir() in Colab is respected
        if root is None:
            root = _resolve_gzgc_root()
        self.root      = Path(root)
        self.ann_file  = Path(ann_file)  if ann_file  else self.root / "annotations" / "instances_train2020.json"
        self.image_dir = Path(image_dir) if image_dir else self.root / "images" / "train2020"
        self.split     = split
        self.transform = transform
        self.individual_id_offset = individual_id_offset

        # ── Load COCO annotations ────────────────────────────────────────────
        with open(self.ann_file) as f:
            coco = json.load(f)

        # image_id → file_name
        id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

        # Group annotations by individual name (zebra only)
        name_to_samples: dict[str, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("category_id") != GZGC_CATEGORY_ID_ZEBRA:
                continue
            fname = id_to_fname.get(ann["image_id"])
            if fname is None:
                continue
            name_to_samples[ann["name"]].append({
                "file_path": self.image_dir / fname,
                "bbox":      ann.get("bbox"),          # [x, y, w, h]
                "viewpoint": ann.get("viewpoint", ""),
                "ann_id":    ann["id"],
            })

        # ── Filter by min images ─────────────────────────────────────────────
        name_to_samples = {
            k: v for k, v in name_to_samples.items()
            if len(v) >= min_images_per_individual
        }

        # ── Stratified individual-level split ────────────────────────────────
        all_names = sorted(name_to_samples.keys())
        rng = random.Random(split_seed)
        rng.shuffle(all_names)

        n       = len(all_names)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        split_map = {
            "train": all_names[:n_train],
            "val":   all_names[n_train : n_train + n_val],
            "test":  all_names[n_train + n_val :],
        }
        selected = split_map[split]

        # ── Assign globally-stable integer IDs ────────────────────────────────
        # Use position in the FULL sorted name list so that every split
        # assigns non-overlapping IDs — train gets 0..n_train-1,
        # val gets n_train..n_train+n_val-1, etc.
        all_names_sorted = sorted(name_to_samples.keys())
        global_name_to_int: dict[str, int] = {
            name: (i + individual_id_offset)
            for i, name in enumerate(all_names_sorted)
        }

        # ── Build flat sample list ───────────────────────────────────────────
        self.samples: list[dict] = []
        for name in selected:
            gid = global_name_to_int[name]
            for s in name_to_samples[name]:
                self.samples.append({
                    "file_path":      s["file_path"],
                    "bbox":           s["bbox"],
                    "viewpoint":      s["viewpoint"],
                    "individual_id":  gid,
                    "individual_name": name,
                    "population_label": POP_A,
                })

        self.num_individuals = len(selected)
        self._individual_ids = sorted(global_name_to_int[n] for n in selected)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        s = self.samples[idx]
        img = Image.open(s["file_path"]).convert("RGB")

        # Crop to annotation bounding box
        if s["bbox"] is not None:
            x, y, w, h = [int(v) for v in s["bbox"]]
            if w > 0 and h > 0:
                img = img.crop((x, y, x + w, y + h))

        if self.transform:
            img = self.transform(img)
        return img, s["individual_id"], s["population_label"]

    @property
    def individual_ids(self) -> list[int]:
        return self._individual_ids

    def stats(self) -> dict:
        return {
            "dataset":     "GZGC",
            "split":       self.split,
            "n_individuals": self.num_individuals,
            "n_samples":   len(self.samples),
            "population":  "plains_zebra (A)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Population B — Labeled Mpala Grevy's Zebra
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_grevys_root() -> Path:
    """Resolve Grevy's dataset root at call-time so os.chdir() in Colab is respected."""
    for p in [Path("labeled_mpala_grevys"), Path("data/labeled_mpala_grevys")]:
        if p.exists():
            return p
    return Path("labeled_mpala_grevys")



class GrevysDataset(Dataset):
    """
    Labeled Mpala Grevy's Zebra dataset loader.

    Layout:
        labeled_mpala_grevys/
            <INDIV_ID>/          ← folder name is the integer individual ID
                <ANNOT_ID>.png   ← one PNG per annotation/crop

    Stats:
      173 unique individuals · 685 images · mean 4.0 images/individual
      (Grevy's zebra is an endangered species — Lewa/Mpala field station data)

    Args:
        root:       Path to labeled_mpala_grevys/. Defaults to 'labeled_mpala_grevys'.
        split:      'train' | 'val' | 'test'. Stratified by individual.
        split_seed: Random seed.
        train_ratio / val_ratio: Fraction of individuals per split.
        transform:  torchvision-compatible transform.
        min_images_per_individual: Drop individuals with fewer images (default 1).
        individual_id_offset: Added to all individual IDs for global uniqueness
                    when combined with GZGC dataset.
    """

    def __init__(
        self,
        root:           Optional[str | Path] = None,
        split:          Split = "train",
        split_seed:     int = 42,
        train_ratio:    float = 0.70,
        val_ratio:      float = 0.15,
        transform:      Optional[Callable] = None,
        min_images_per_individual: int = 1,
        individual_id_offset: int = 0,
    ) -> None:
        super().__init__()
        # Resolve root lazily so os.chdir() in Colab is respected
        if root is None:
            root = _resolve_grevys_root()
        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.individual_id_offset = individual_id_offset

        # ── Discover individuals from folder structure ────────────────────────
        local_id_to_images: dict[int, list[Path]] = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            local_id = int(folder.name)
            pngs = sorted(folder.glob("*.png"))
            if len(pngs) >= min_images_per_individual:
                local_id_to_images[local_id] = pngs

        # ── Stratified individual-level split ────────────────────────────────
        all_local_ids = sorted(local_id_to_images.keys())
        rng = random.Random(split_seed)
        rng.shuffle(all_local_ids)

        n       = len(all_local_ids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        split_map = {
            "train": all_local_ids[:n_train],
            "val":   all_local_ids[n_train : n_train + n_val],
            "test":  all_local_ids[n_train + n_val :],
        }
        selected_local_ids = split_map[split]

        # ── Assign globally-stable integer IDs ───────────────────────────────
        # Use position in FULL sorted local_id list so IDs are stable
        # across splits and non-overlapping.
        all_local_sorted = sorted(local_id_to_images.keys())
        global_local_to_global: dict[int, int] = {
            local_id: (i + individual_id_offset)
            for i, local_id in enumerate(all_local_sorted)
        }

        # ── Build flat sample list ───────────────────────────────────────────
        self.samples: list[dict] = []
        for local_id in selected_local_ids:
            gid = global_local_to_global[local_id]
            for img_path in local_id_to_images[local_id]:
                self.samples.append({
                    "file_path":       img_path,
                    "individual_id":   gid,
                    "local_indiv_id":  local_id,
                    "population_label": POP_B,
                })

        self.num_individuals = len(selected_local_ids)
        self._individual_ids = sorted(global_local_to_global[lid] for lid in selected_local_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        s = self.samples[idx]
        img = Image.open(s["file_path"]).convert("RGB")
        # Images are already cropped zebra patches — no bbox needed
        if self.transform:
            img = self.transform(img)
        return img, s["individual_id"], s["population_label"]

    @property
    def individual_ids(self) -> list[int]:
        return self._individual_ids

    def stats(self) -> dict:
        return {
            "dataset":       "Labeled Mpala Grevy's Zebra",
            "split":         self.split,
            "n_individuals": self.num_individuals,
            "n_samples":     len(self.samples),
            "population":    "grevys_zebra (B)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build both datasets with matching ID spaces
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    split: Split,
    transform: Optional[Callable],
    split_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    gzgc_root:   Optional[str | Path] = None,
    grevys_root: Optional[str | Path] = None,
) -> tuple["GZGCDataset", "GrevysDataset"]:
    """
    Build GZGC (Pop A) and Grevy's (Pop B) datasets for the given split.
    Individual IDs are globally unique across both populations AND across all
    train/val/test splits.

    ID space layout:
      GZGC     (Pop A): 0 .. n_gzgc_total - 1
      Grevy's  (Pop B): n_gzgc_total .. n_gzgc_total + n_grevys_total - 1

    Returns:
        (ds_a, ds_b) ready to be wrapped in CombinedZebraDataset.
    """
    # Resolve lazily so CWD set by os.chdir() in Colab is honoured
    if gzgc_root is None:
        gzgc_root = _resolve_gzgc_root()
    if grevys_root is None:
        grevys_root = _resolve_grevys_root()

    split_kw = dict(split_seed=split_seed, train_ratio=train_ratio, val_ratio=val_ratio)

    ds_a = GZGCDataset(root=gzgc_root, split=split, transform=transform, **split_kw)

    # Count total GZGC individuals across ALL splits to set the Grevy's offset.
    import json as _json
    _ann = ds_a.ann_file
    with open(_ann) as _f:
        _coco = _json.load(_f)
    _names = {a["name"] for a in _coco["annotations"]
              if a.get("category_id") == GZGC_CATEGORY_ID_ZEBRA}
    _n_gzgc_total = len(_names)  # total unique zebra individuals in dataset

    ds_b = GrevysDataset(
        root=grevys_root, split=split, transform=transform,
        individual_id_offset=_n_gzgc_total,   # starts after ALL GZGC individuals
        **split_kw,
    )
    return ds_a, ds_b

