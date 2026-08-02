"""
zebraid/models/evaluate_compression.py
Accuracy-vs-Z-Hash-size trade-off benchmark.

Sweeps across code sizes (128b, 256b, 512b) and backends (pca_binarize, faiss_pq),
reports Rank-1 and Rank-5 accuracy versus the full-embedding baseline.

Run after training:
    python -m zebraid.models.evaluate_compression \\
        --checkpoint checkpoints/zebraid/megadescriptor/seed42/best_model.pt \\
        --config configs/default.yaml

Output: results/compression_tradeoff.csv  (used for Figure in the paper)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from zebraid.data.dataset import ZebraDataset, POP_A, POP_B
from zebraid.data.transforms import eval_transforms
from zebraid.models.backbone import build_embedder, ZebraEmbedder
from zebraid.models.zhash import ZHashEncoder
from zebraid.models.evaluate import compute_cmc_map


def _extract_embeddings(
    model: ZebraEmbedder,
    dataset: ZebraDataset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all embeddings and labels from a dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, individual_ids, _ in loader:
            embs = model(images.to(device)).cpu().numpy()
            embeddings_list.append(embs)
            labels_list.extend(individual_ids.tolist())

    return np.concatenate(embeddings_list, axis=0), np.array(labels_list)


def _rank_accuracy_from_codes(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    top_k: int = 5,
) -> dict:
    """Compute Rank-1 and Rank-k using Hamming distance between binary codes."""
    N_q = len(test_codes)
    rank1 = 0
    rankk = 0

    for i in range(N_q):
        query = test_codes[i]
        # Hamming distance via XOR and popcount
        xor = np.bitwise_xor(train_codes, query[None, :])
        hamming = np.unpackbits(xor, axis=1).sum(axis=1)
        sorted_idx = np.argsort(hamming)
        sorted_labels = train_labels[sorted_idx]

        if sorted_labels[0] == test_labels[i]:
            rank1 += 1
        if (sorted_labels[:top_k] == test_labels[i]).any():
            rankk += 1

    return {
        "rank1": rank1 / N_q,
        f"rank{top_k}": rankk / N_q,
    }


def run_compression_benchmark(
    checkpoint_path: str,
    config_path: str = "configs/default.yaml",
    output_path: str = "results/compression_tradeoff.csv",
    split_seed: int = 42,
) -> None:
    import yaml
    cfg = yaml.safe_load(open(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = "megadescriptor"  # run on primary backbone
    img_size = 384

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_embedder(backbone_name=backbone_name,
                           embedding_dim=cfg["model"]["embedding_dim"],
                           pretrained=False, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ── Load datasets ────────────────────────────────────────────────────────
    data_root_a = Path(cfg["paths"]["data_root"]) / cfg["datasets"]["population_a"]["name"]
    ann_a = data_root_a / "annotations" / "instances.json"
    t_eval = eval_transforms(img_size)
    split_kw = dict(split_seed=split_seed,
                    train_ratio=cfg["datasets"]["train_ratio"],
                    val_ratio=cfg["datasets"]["val_ratio"])

    ds_train = ZebraDataset(data_root_a, ann_a, POP_A, "train", transform=t_eval, **split_kw)
    ds_val   = ZebraDataset(data_root_a, ann_a, POP_A, "val",   transform=t_eval, **split_kw)

    print("[compression] Extracting embeddings ...")
    train_embs, train_labels = _extract_embeddings(model, ds_train, device)
    val_embs,   val_labels   = _extract_embeddings(model, ds_val,   device)

    # ── Baseline: full embedding ─────────────────────────────────────────────
    baseline = compute_cmc_map(model, ds_val, device, top_k=5)
    print(f"[compression] Baseline (full emb): R1={baseline['rank1']:.4f} mAP={baseline['map']:.4f}")

    # ── Sweep ────────────────────────────────────────────────────────────────
    rows = []
    sizes = cfg["zhash"]["benchmark_sizes_bits"]
    backends: list[str] = ["pca_binarize", "faiss_pq"]

    for backend in backends:
        for size_bits in sizes:
            print(f"[compression] backend={backend} size={size_bits}b ...")
            try:
                enc = ZHashEncoder(size_bits=size_bits, backend=backend)
                enc.fit(train_embs)

                train_codes = enc.encode_batch(train_embs)
                val_codes   = enc.encode_batch(val_embs)

                metrics = _rank_accuracy_from_codes(
                    train_codes, val_codes, train_labels, val_labels, top_k=5
                )

                rows.append({
                    "backend": backend,
                    "size_bits": size_bits,
                    "size_bytes": size_bits // 8,
                    "rank1": round(metrics["rank1"], 4),
                    "rank5": round(metrics.get("rank5", float("nan")), 4),
                    "rank1_drop_vs_baseline": round(baseline["rank1"] - metrics["rank1"], 4),
                })
            except Exception as e:
                print(f"  [SKIP] {backend}/{size_bits}b failed: {e}")

    # Add baseline row
    rows.insert(0, {
        "backend": "full_embedding",
        "size_bits": cfg["model"]["embedding_dim"] * 32,  # float32
        "size_bytes": cfg["model"]["embedding_dim"] * 4,
        "rank1": round(baseline["rank1"], 4),
        "rank5": round(baseline.get("rank5", float("nan")), 4),
        "rank1_drop_vs_baseline": 0.0,
    })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["backend", "size_bits", "size_bytes", "rank1", "rank5", "rank1_drop_vs_baseline"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[compression] Results saved to {output_path}")
    for row in rows:
        print(f"  {row['backend']:20s} {row['size_bits']:5d}b → R1={row['rank1']:.4f}  "
              f"drop={row['rank1_drop_vs_baseline']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/compression_tradeoff.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_compression_benchmark(args.checkpoint, args.config, args.output, args.seed)
