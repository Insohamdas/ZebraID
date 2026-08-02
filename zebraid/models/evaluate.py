"""
zebraid/models/evaluate.py
Evaluation utilities: CMC curve and mAP for re-identification.

compute_cmc_map():
  - Builds a gallery from the evaluation dataset.
  - For each query, computes cosine similarity against all gallery items.
  - Reports Rank-1, Rank-5 accuracy and mean Average Precision (mAP).

These are the standard re-id evaluation metrics used by all comparable
systems (HotSpotter, Wildbook, WildlifeDatasets benchmarks).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from zebraid.data.dataset import ZebraDataset
from zebraid.models.backbone import ZebraEmbedder


def compute_cmc_map(
    model: ZebraEmbedder,
    dataset: ZebraDataset,
    device: torch.device,
    top_k: int = 5,
    batch_size: int = 32,
) -> dict:
    """
    Compute CMC (Rank-1, Rank-k) and mAP on a ZebraDataset.

    The evaluation follows the standard re-id protocol:
      - Gallery = all samples in the dataset.
      - Query   = same dataset (leave-one-out: query item excluded from gallery).

    Args:
        model:      Trained ZebraEmbedder in eval mode.
        dataset:    Evaluation ZebraDataset.
        device:     Torch device.
        top_k:      Rank-k accuracy to compute (default 5).
        batch_size: Batch size for embedding extraction.

    Returns:
        Dict with keys: 'rank1', 'rank5' (or 'rankk'), 'map'.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, individual_ids, _ in loader:
            images = images.to(device)
            embs = model(images).cpu().numpy()
            all_embeddings.append(embs)
            all_labels.extend(individual_ids.tolist())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
    labels = np.array(all_labels)                         # (N,)

    N = len(labels)
    rank1_correct = 0
    rankk_correct = 0
    ap_sum = 0.0

    for i in range(N):
        query_emb = embeddings[i]         # (D,)
        query_label = labels[i]

        # Cosine similarity against all gallery items (embeddings are L2-normalized)
        sims = embeddings @ query_emb     # (N,)
        sims[i] = -1.0                    # exclude query itself

        sorted_idx = np.argsort(-sims)    # descending
        sorted_labels = labels[sorted_idx]

        # CMC
        matches = sorted_labels == query_label
        if matches[0]:
            rank1_correct += 1
        if matches[:top_k].any():
            rankk_correct += 1

        # Average Precision
        match_positions = np.where(matches)[0] + 1  # 1-indexed
        if len(match_positions) > 0:
            precision_at_k = [
                (j + 1) / pos for j, pos in enumerate(match_positions)
            ]
            ap_sum += np.mean(precision_at_k)

    return {
        "rank1":         rank1_correct / N,
        "rank5":         rankk_correct / N,   # alias; equals rank{top_k} when top_k=5
        f"rank{top_k}":  rankk_correct / N,
        "map":           ap_sum / N,
        "n_queries":     N,
    }

