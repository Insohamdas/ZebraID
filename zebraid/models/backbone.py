"""
zebraid/models/backbone.py
ZebraEmbedder — backbone wrapper for zebra stripe embedding.

Supports two backbones:
  1. MegaDescriptor-L-384 (primary): wildlife-pretrained via the
     wildlife-datasets toolkit. Strongest starting point for stripe patterns.
  2. timm ResNet50 (ablation): standard ImageNet-pretrained baseline,
     used to demonstrate the value of the wildlife-specific pretraining.

Both produce L2-normalized embeddings of configurable dimension.
The ZHashEncoder (zebraid/models/zhash.py) then compresses these to binary codes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Backbone IDs ──────────────────────────────────────────────────────────────
MEGADESCRIPTOR_MODEL_ID = "hf-hub:BVRA/MegaDescriptor-L-384"
RESNET50_MODEL_ID = "resnet50"


class ZebraEmbedder(nn.Module):
    """
    Wraps a pre-trained backbone and adds:
      - A linear projection head (backbone_dim → embedding_dim).
      - L2 normalization of the output embedding.

    Args:
        backbone_name: Either 'megadescriptor' or 'resnet50'.
        embedding_dim: Output embedding dimension (default 512).
        pretrained:    Whether to load pretrained weights (default True).
    """

    def __init__(
        self,
        backbone_name: str = "megadescriptor",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        # ── Load backbone ────────────────────────────────────────────────────
        if backbone_name == "megadescriptor":
            # MegaDescriptor is distributed as a timm-compatible model on HuggingFace
            print("Loading MegaDescriptor model...", flush=True)
            try:
                self.backbone = timm.create_model(
                    MEGADESCRIPTOR_MODEL_ID,
                    pretrained=pretrained,
                    num_classes=0,      # remove classification head; use feature output
                )
                print("MegaDescriptor loaded.", flush=True)
            except Exception as e:
                # Network or HF hub issues can cause timm to fail. Fall back to a
                # lightweight local backbone (ResNet50 from torchvision) to allow
                # training to proceed in restricted/offline environments.
                print(f"Warning: failed to load MegaDescriptor via timm ({e}); falling back to local ResNet50.", flush=True)
                try:
                    import torchvision.models as tv
                    local = tv.resnet50(weights=None)
                    # Replace classification head with identity to get features
                    in_feat = local.fc.in_features
                    local.fc = nn.Identity()
                    # Mimic timm's num_features attribute
                    local.num_features = in_feat
                    self.backbone = local
                    print("Fallback ResNet50 loaded.", flush=True)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load fallback backbone: {e2}") from e2
            backbone_out_dim = self.backbone.num_features
        elif backbone_name == "resnet50":
            self.backbone = timm.create_model(
                RESNET50_MODEL_ID,
                pretrained=pretrained,
                num_classes=0,      # remove classification head
                global_pool="avg",  # global average pooling
            )
            backbone_out_dim = self.backbone.num_features
        else:
            raise ValueError(
                f"backbone_name must be 'megadescriptor' or 'resnet50', got '{backbone_name}'"
            )

        # ── Projection head ──────────────────────────────────────────────────
        # A small 2-layer MLP projects from backbone_out_dim → embedding_dim.
        # BatchNorm before final layer stabilizes triplet loss training.
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.BatchNorm1d(backbone_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W).
        Returns:
            L2-normalized embedding of shape (B, embedding_dim).
        """
        features = self.backbone(x)          # (B, backbone_out_dim)
        projected = self.projector(features)  # (B, embedding_dim)
        return F.normalize(projected, p=2, dim=1)  # L2 normalize

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — used in inference scripts."""
        return self.forward(x)


def build_embedder(
    backbone_name: str = "megadescriptor",
    embedding_dim: int = 512,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> ZebraEmbedder:
    """
    Factory function to build and move a ZebraEmbedder to the target device.

    Args:
        backbone_name: 'megadescriptor' or 'resnet50'.
        embedding_dim: Output embedding dimension.
        pretrained:    Load pretrained weights.
        device:        Target device (defaults to CUDA if available, else CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZebraEmbedder(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    ).to(device)

    return model
