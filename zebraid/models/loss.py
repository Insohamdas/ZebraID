"""
zebraid/models/loss.py
Loss functions for ZebraID metric learning.

Provides:
  - TripletLossWithMining: triplet loss with online hard-negative mining via
    pytorch-metric-learning. This is the primary loss function.
  - ArcFaceLoss: ArcFace margin loss for ablation / multi-task training.

Both losses operate on L2-normalized embeddings (as produced by ZebraEmbedder).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from pytorch_metric_learning import losses, miners
    PML_AVAILABLE = True
except ImportError:
    PML_AVAILABLE = False


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with online hard-negative mining.

    Uses pytorch-metric-learning's TripletMarginLoss with BatchHardMiner,
    which selects the hardest positive and hardest negative per anchor within
    each batch — more efficient than offline triplet sampling.

    Args:
        margin:        Triplet margin (default 0.3).
        mining_type:   One of 'hard', 'semi_hard', 'all' (default 'hard').
    """

    def __init__(self, margin: float = 0.3, mining_type: str = "hard") -> None:
        super().__init__()
        if not PML_AVAILABLE:
            raise ImportError(
                "pytorch-metric-learning is required for TripletLossWithMining. "
                "Install it with: pip install pytorch-metric-learning"
            )

        self.loss_fn = losses.TripletMarginLoss(margin=margin)

        if mining_type == "hard":
            self.miner = miners.BatchHardMiner()
        elif mining_type == "semi_hard":
            self.miner = miners.BatchEasyHardMiner(neg_strategy="semihard")
        elif mining_type == "all":
            self.miner = miners.BatchEasyHardMiner(neg_strategy="easy")
        else:
            raise ValueError(f"Unknown mining_type: {mining_type}")

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings, shape (B, D).
            labels:     Individual ID labels, shape (B,).
        Returns:
            Scalar triplet loss.
        """
        hard_pairs = self.miner(embeddings, labels)
        return self.loss_fn(embeddings, labels, hard_pairs)


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss for ablation / multi-task training.

    ArcFace treats each individual as a class and optimizes angular decision
    boundaries in the embedding space. It's complementary to triplet loss and
    can be used jointly (loss = triplet + λ * arcface).

    Args:
        num_classes:   Number of individual identities in the training set.
        embedding_dim: Dimensionality of the embedding (must match ZebraEmbedder).
        scale:         ArcFace scale parameter s (default 64).
        margin:        ArcFace angular margin m in radians (default 0.5).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        if not PML_AVAILABLE:
            raise ImportError(
                "pytorch-metric-learning is required for ArcFaceLoss. "
                "Install it with: pip install pytorch-metric-learning"
            )

        self.loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin,
            scale=scale,
        )

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings, shape (B, D).
            labels:     Individual ID labels, shape (B,).
        Returns:
            Scalar ArcFace loss.
        """
        return self.loss_fn(embeddings, labels)
