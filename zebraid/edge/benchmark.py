"""
zebraid/edge/benchmark.py
Edge inference benchmark — CPU-proxy simulation on Mac mini.

⚠️  PROXY MEASUREMENT: This script runs on Mac mini CPU via ONNX Runtime.
    Numbers are clearly labeled as proxy measurements.
    Re-run on Raspberry Pi 5 / Jetson Orin Nano when hardware is collected.

Measures:
  - Inference latency (ms per image, mean ± std over N trials).
  - Model size on disk (MB) for fp32 and int8.
  - Z-Hash payload size (bytes) — hardware-independent; this is what
    would be transmitted over LoRaWAN from a field station.

Output: results/edge_benchmark_mac_proxy.json

Usage:
    python -m zebraid.edge.benchmark \\
        --onnx_fp32 checkpoints/onnx/zebraid_megadescriptor_fp32.onnx \\
        --onnx_int8 checkpoints/onnx/zebraid_megadescriptor_int8.onnx \\
        --zhash_size_bytes 32
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

# ── Platform note — written into every result file ────────────────────────────
PLATFORM_NOTE = (
    "PROXY MEASUREMENT on Mac mini CPU via ONNX Runtime. "
    "Re-run on Raspberry Pi 5 / Jetson Orin Nano when hardware is collected."
)


def benchmark_onnx(
    onnx_path: str,
    img_size: int = 384,
    n_trials: int = 100,
    zhash_size_bytes: int = 32,
) -> dict:
    """
    Run inference latency benchmark on an ONNX model.

    Args:
        onnx_path:        Path to the .onnx file.
        img_size:         Input image size (height = width).
        n_trials:         Number of inference runs to average.
        zhash_size_bytes: Size of the Z-Hash payload in bytes (hardware-independent).
    Returns:
        Dict of benchmark metrics.
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime is required: pip install onnxruntime")

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    input_name  = sess.get_inputs()[0].name

    # Warm up
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)

    model_size_mb = Path(onnx_path).stat().st_size / 1e6

    return {
        "onnx_path": onnx_path,
        "model_size_mb": round(model_size_mb, 2),
        "n_trials": n_trials,
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "std_latency_ms":  round(float(np.std(latencies)),  3),
        "p50_latency_ms":  round(float(np.percentile(latencies, 50)), 3),
        "p99_latency_ms":  round(float(np.percentile(latencies, 99)), 3),
        "zhash_payload_bytes": zhash_size_bytes,
        "zhash_payload_bits":  zhash_size_bytes * 8,
        "lora_tx_feasibility": (
            "YES — LoRaWAN maximum payload is 51–222 bytes per frame (SF7-SF12); "
            f"{zhash_size_bytes}B Z-Hash fits in a single frame."
            if zhash_size_bytes <= 51
            else f"MARGINAL — {zhash_size_bytes}B may require SF7 or packet fragmentation."
        ),
        "platform": platform.platform(),
        "ort_version": ort.__version__,
        "proxy_note": PLATFORM_NOTE,
    }


def run_benchmark(
    onnx_fp32_path: Optional[str],
    onnx_int8_path: Optional[str] = None,
    img_size: int = 384,
    n_trials: int = 100,
    zhash_size_bytes: int = 32,
    output_path: str = "results/edge_benchmark_mac_proxy.json",
) -> dict:
    from typing import Optional

    results = {
        "proxy_note": PLATFORM_NOTE,
        "platform": platform.platform(),
    }

    if onnx_fp32_path:
        print(f"[edge_benchmark] Benchmarking fp32: {onnx_fp32_path}")
        results["fp32"] = benchmark_onnx(onnx_fp32_path, img_size, n_trials, zhash_size_bytes)
        print(
            f"  latency: {results['fp32']['mean_latency_ms']:.1f} ± "
            f"{results['fp32']['std_latency_ms']:.1f} ms  "
            f"model: {results['fp32']['model_size_mb']:.1f} MB  "
            f"payload: {zhash_size_bytes}B"
        )

    if onnx_int8_path:
        print(f"[edge_benchmark] Benchmarking int8: {onnx_int8_path}")
        results["int8"] = benchmark_onnx(onnx_int8_path, img_size, n_trials, zhash_size_bytes)
        print(
            f"  latency: {results['int8']['mean_latency_ms']:.1f} ± "
            f"{results['int8']['std_latency_ms']:.1f} ms  "
            f"model: {results['int8']['model_size_mb']:.1f} MB"
        )

    if onnx_fp32_path and onnx_int8_path:
        speedup = results["fp32"]["mean_latency_ms"] / results["int8"]["mean_latency_ms"]
        size_reduction = results["fp32"]["model_size_mb"] / results["int8"]["model_size_mb"]
        results["int8_vs_fp32"] = {
            "speedup": round(speedup, 2),
            "size_reduction": round(size_reduction, 2),
        }
        print(f"  INT8 speedup: {speedup:.2f}×  size reduction: {size_reduction:.2f}×")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[edge_benchmark] Results saved to {output_path}")
    print(f"[edge_benchmark] *** {PLATFORM_NOTE} ***")
    return results


if __name__ == "__main__":
    from typing import Optional
    parser = argparse.ArgumentParser(description="ZebraID edge inference benchmark (CPU proxy)")
    parser.add_argument("--onnx_fp32",  default=None)
    parser.add_argument("--onnx_int8",  default=None)
    parser.add_argument("--img_size",   type=int, default=384)
    parser.add_argument("--n_trials",   type=int, default=100)
    parser.add_argument("--zhash_size_bytes", type=int, default=32)
    parser.add_argument("--output",     default="results/edge_benchmark_mac_proxy.json")
    args = parser.parse_args()

    run_benchmark(
        onnx_fp32_path=args.onnx_fp32,
        onnx_int8_path=args.onnx_int8,
        img_size=args.img_size,
        n_trials=args.n_trials,
        zhash_size_bytes=args.zhash_size_bytes,
        output_path=args.output,
    )
