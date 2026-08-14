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
from pathlib import Path
import shutil
import os
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from PIL import Image

from zebraid.data.dataset import ZebraDataset
from zebraid.models.backbone import ZebraEmbedder


def compute_cmc_map(
    model: ZebraEmbedder,
    dataset: ZebraDataset,
    device: torch.device,
    top_k: int = 5,
    batch_size: int = 32,
    save_examples_dir: str | Path | None = None,
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
        save_examples_dir: Optional directory to save visual retrieval examples.

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

    # ── Evaluation Safety Check ──────────────────────────────────────────────
    if not np.isfinite(embeddings).all():
        nan_count = int(np.isnan(embeddings).sum())
        inf_count = int(np.isinf(embeddings).sum())
        raise RuntimeError(
            f"Evaluation Safety Check Failed: Extracted embeddings contain non-finite values "
            f"({nan_count} NaNs, {inf_count} Infs out of {embeddings.size} total elements). "
            f"Model weights or intermediate activations are corrupted."
        )

    from collections import Counter
    label_counts = Counter(labels)

    N = len(labels)
    rank1_correct_all = 0
    rankk_correct_all = 0
    ap_sum_all = 0.0

    rank1_correct_multi = 0
    rankk_correct_multi = 0
    ap_sum_multi = 0.0
    n_queries_multi = 0

    for i in range(N):
        query_emb = embeddings[i]         # (D,)
        query_label = labels[i]
        is_multi = (label_counts[query_label] > 1)

        # Cosine similarity against all gallery items (embeddings are L2-normalized)
        sims = embeddings @ query_emb     # (N,)
        sims[i] = -1.0                    # exclude query itself

        if not np.isfinite(sims).all():
            raise RuntimeError(
                f"Evaluation Safety Check Failed: Cosine similarity vector for query index {i} "
                f"contains NaN or Inf values."
            )

        sorted_idx = np.argsort(-sims)    # descending
        sorted_labels = labels[sorted_idx]

        # CMC
        matches = sorted_labels == query_label
        is_r1 = bool(matches[0])
        is_rk = bool(matches[:top_k].any())

        if is_r1:
            rank1_correct_all += 1
        if is_rk:
            rankk_correct_all += 1

        # Average Precision
        match_positions = np.where(matches)[0] + 1  # 1-indexed
        if len(match_positions) > 0:
            precision_at_k = [
                (j + 1) / pos for j, pos in enumerate(match_positions)
            ]
            ap = float(np.mean(precision_at_k))
        else:
            ap = 0.0
        ap_sum_all += ap

        if is_multi:
            n_queries_multi += 1
            if is_r1:
                rank1_correct_multi += 1
            if is_rk:
                rankk_correct_multi += 1
            ap_sum_multi += ap

        # Save retrieval examples
        if save_examples_dir is not None and i < 20 and plt is not None:
            out_dir = Path(save_examples_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
            
            # Query image
            q_sample = dataset.samples[i]
            q_img = Image.open(q_sample["file_path"]).convert("RGB")
            if q_sample.get("bbox") is not None:
                x, y, w, h = [int(v) for v in q_sample["bbox"]]
                q_img = q_img.crop((x, y, x + w, y + h))
            
            axes[0].imshow(q_img)
            axes[0].set_title(f"Query\nID: {query_label}")
            axes[0].axis("off")
            
            # Top-k retrieved images
            for k in range(top_k):
                ret_idx = sorted_idx[k]
                ret_label = sorted_labels[k]
                ret_sample = dataset.samples[ret_idx]
                
                r_img = Image.open(ret_sample["file_path"]).convert("RGB")
                if ret_sample.get("bbox") is not None:
                    x, y, w, h = [int(v) for v in ret_sample["bbox"]]
                    r_img = r_img.crop((x, y, x + w, y + h))
                    
                axes[k+1].imshow(r_img)
                color = "green" if ret_label == query_label else "red"
                axes[k+1].set_title(f"Rank {k+1}\nID: {ret_label}", color=color)
                
                # Add border based on correctness
                for spine in axes[k+1].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                axes[k+1].set_xticks([])
                axes[k+1].set_yticks([])
                
            plt.tight_layout()
            # Prefix with SUCCESS or FAIL based on rank-1
            prefix = "SUCCESS" if matches[0] else "FAIL"
            plt.savefig(out_dir / f"{prefix}_query_{i}_id_{query_label}.png", bbox_inches='tight')
            plt.close()

    n_singletons = N - n_queries_multi

    return {
        "rank1":              rank1_correct_all / N if N > 0 else 0.0,
        "rank5":              rankk_correct_all / N if N > 0 else 0.0,
        f"rank{top_k}":       rankk_correct_all / N if N > 0 else 0.0,
        "map":                ap_sum_all / N if N > 0 else 0.0,
        "n_queries":          N,
        "rank1_multi":        rank1_correct_multi / n_queries_multi if n_queries_multi > 0 else 0.0,
        "rank5_multi":        rankk_correct_multi / n_queries_multi if n_queries_multi > 0 else 0.0,
        f"rank{top_k}_multi": rankk_correct_multi / n_queries_multi if n_queries_multi > 0 else 0.0,
        "map_multi":          ap_sum_multi / n_queries_multi if n_queries_multi > 0 else 0.0,
        "n_queries_multi":    n_queries_multi,
        "n_singletons":       n_singletons,
    }

