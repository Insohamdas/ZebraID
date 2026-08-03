"""
zebraid/data/transforms.py
Augmentation pipelines for ZebraID training and evaluation.

Design choices:
  - train_transforms:  heavy augmentation simulating field conditions
    (lighting variation, mud/vegetation occlusion via random erasing,
    motion blur, partial-flank crops).
  - eval_transforms:   deterministic resize + normalize only.
  - Both pipelines output tensors compatible with torchvision models
    (ImageNet mean/std normalization).
"""

from __future__ import annotations

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# MegaDescriptor-L uses 384×384 input; ResNet50 uses 224×224.
# We default to 384 so both models can share the same pipeline —
# ResNet50 will internally downsample, which is fine for an ablation.
DEFAULT_IMG_SIZE = 384


def train_transforms(img_size: int = DEFAULT_IMG_SIZE) -> T.Compose:
    """
    Augmentation pipeline for training.

    Simulates realistic field conditions:
      - RandomResizedCrop: partial-flank shots and varying distances.
      - RandomHorizontalFlip: zebra can face either direction.
      - ColorJitter: variable lighting, time-of-day, camera white balance.
      - GaussianBlur: motion blur from moving animals or camera shake.
      - RandomErasing: mud patches, vegetation occlusion, fence obscuring stripes.
      - Normalize: ImageNet stats (used by both MegaDescriptor and timm ResNet).
    """
    return T.Compose(
        [
            T.RandomResizedCrop(
                img_size,
                scale=(0.5, 1.0),        # allow aggressive crops
                ratio=(0.75, 1.33),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.05,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # RandomErasing simulates mud / vegetation / fence occlusion
            T.RandomErasing(
                p=0.4,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0,               # erase to black (neutral)
                inplace=False,
            ),
        ]
    )


def eval_transforms(img_size: int = DEFAULT_IMG_SIZE) -> T.Compose:
    """
    Deterministic evaluation pipeline.
    Resize to slightly larger than target, then centre-crop for consistency.
    """
    resize_size = int(img_size * 1.143)  # ≈ 1/0.875 ratio, standard practice
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
