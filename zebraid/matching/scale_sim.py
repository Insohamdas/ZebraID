"""
zebraid/matching/scale_sim.py
FAISS scale simulation benchmark.

Augments real Z-Hash embeddings with synthetic "phantom" vectors (sampled
from the same distribution) to populate the index at 1k / 100k / 1M entries.
Benchmarks Recall@1, Recall@5, query latency, and memory footprint.

⚠️  All outputs are explicitly labeled "SIMULATED-SCALE BENCHMARK".
    No claim is made about a real zebra population of this size.

Run:
    python -m zebraid.matching.scale_sim \\
        --zhash_file results/val_zhashes.npy \\
        --labels_file results/val_labels.npy \\
        --output results/scale_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _memory_usage_mb() -> float:
    """Return current process RSS in MB (macOS/Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        return float("nan")


def run_scale_simulation(
    real_codes: np.ndarray,      # (N_real, code_bytes) uint8
    real_labels: np.ndarray,     # (N_real,) int
    scale_sizes: list[int],
    n_queries: int = 100,
    top_k: int = 5,
    output_path: str = "results/scale_sim.csv",
    index_types: list[str] = ("flat_binary", "ivf_pq"),
) -> list[dict]:
    """
    For each scale in scale_sizes:
      1. Create a FAISS index.
      2. Add real codes + phantom codes (sampled from real distribution) until
         the target scale is reached.
      3. Run n_queries queries (from real codes, so we can check recall).
      4. Report Recall@1, Recall@5, mean query latency, and memory.

    Returns list of result dicts (also written to output_path CSV).
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu")

    N_real, code_bytes = real_codes.shape
    n_bits = code_bytes * 8

    # ── Random query selection from real data ────────────────────────────────
    rng = np.random.default_rng(42)
    query_idx = rng.choice(N_real, size=min(n_queries, N_real), replace=False)
    query_codes  = real_codes[query_idx]    # (n_q, code_bytes)
    query_labels = real_labels[query_idx]   # (n_q,)

    rows = []

    for scale in scale_sizes:
        for index_type in index_types:
            print(f"\n[scale_sim] scale={scale:,}  index={index_type}")

            # ── Build index ──────────────────────────────────────────────────
            if index_type == "flat_binary":
                index = faiss.IndexBinaryFlat(n_bits)
            elif index_type == "ivf_pq":
                quantizer = faiss.IndexBinaryFlat(n_bits)
                nlist = max(1, int(np.sqrt(scale)))
                index = faiss.IndexBinaryIVF(quantizer, n_bits, nlist)
                index.nprobe = min(10, nlist)
            else:
                raise ValueError(f"Unknown index_type: {index_type}")

            # ── Populate with real + phantom codes ───────────────────────────
            n_phantom = max(0, scale - N_real)
            all_codes = [real_codes]

            if n_phantom > 0:
                # Sample phantom codes from real distribution:
                # randomly combine bits from real codes
                phantom = real_codes[rng.integers(0, N_real, size=n_phantom)]
                # Bit-flip a random ~20% to make them slightly different
                flip_mask = rng.random((n_phantom, code_bytes)) < 0.2
                flip_bytes = flip_mask.astype(np.uint8) * 255
                phantom = np.bitwise_xor(phantom, flip_bytes)
                all_codes.append(phantom.astype(np.uint8))

            gallery = np.concatenate(all_codes, axis=0)[:scale].astype(np.uint8)
            gallery_labels = np.concatenate(
                [real_labels, np.full(n_phantom, -1)]
            )[:scale]

            if index_type == "ivf_pq" and scale >= 100:
                index.train(gallery)

            mem_before = _memory_usage_mb()
            index.add(gallery)
            mem_after = _memory_usage_mb()

            # ── Query benchmark ──────────────────────────────────────────────
            latencies = []
            recall1_hits = 0
            recall5_hits = 0

            for i, (qcode, qlabel) in enumerate(zip(query_codes, query_labels)):
                t0 = time.perf_counter()
                D, I = index.search(qcode.reshape(1, -1), top_k)
                latencies.append((time.perf_counter() - t0) * 1000)

                retrieved_labels = gallery_labels[I[0]]
                if retrieved_labels[0] == qlabel:
                    recall1_hits += 1
                if (retrieved_labels == qlabel).any():
                    recall5_hits += 1

            row = {
                "NOTE": "SIMULATED-SCALE BENCHMARK — not real zebra population",
                "index_type": index_type,
                "scale": scale,
                "n_real": N_real,
                "n_phantom": n_phantom,
                "n_queries": len(query_codes),
                "recall_at_1": round(recall1_hits / len(query_codes), 4),
                f"recall_at_{top_k}": round(recall5_hits / len(query_codes), 4),
                "mean_latency_ms": round(float(np.mean(latencies)), 3),
                "p99_latency_ms":  round(float(np.percentile(latencies, 99)), 3),
                "index_memory_mb": round(mem_after - mem_before, 2),
                "code_size_bytes": code_bytes,
            }
            rows.append(row)
            print(
                f"  R@1={row['recall_at_1']:.4f}  "
                f"R@{top_k}={row[f'recall_at_{top_k}']:.4f}  "
                f"latency={row['mean_latency_ms']:.2f}ms  "
                f"mem≈{row['index_memory_mb']:.1f}MB"
            )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[scale_sim] Results saved to {output_path}")
        print("[scale_sim] *** All results are SIMULATED-SCALE — see NOTE column ***")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZebraID scale simulation benchmark (SIMULATED — not real zebras)"
    )
    parser.add_argument("--zhash_file",  required=False, default=None, help="Path to .npy of Z-Hash codes (uint8)")
    parser.add_argument("--labels_file", required=False, default=None, help="Path to .npy of individual labels (int)")
    parser.add_argument("--output", default="results/scale_sim.csv")
    parser.add_argument("--scales", nargs="+", type=int, default=[1000, 100000, 1000000])
    parser.add_argument("--n_queries", type=int, default=100)
    args = parser.parse_args()

    if args.zhash_file and args.labels_file:
        real_codes  = np.load(args.zhash_file)
        real_labels = np.load(args.labels_file)
    else:
        print("[scale_sim] No input npy files provided — generating 500 synthetic 256-bit Z-Hash vectors...")
        real_codes  = np.random.randint(0, 256, size=(500, 32), dtype=np.uint8)
        real_labels = np.arange(500)

    run_scale_simulation(
        real_codes=real_codes,
        real_labels=real_labels,
        scale_sizes=args.scales,
        n_queries=args.n_queries,
        output_path=args.output,
    )
