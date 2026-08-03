#!/usr/bin/env python3
"""
scripts/run_mac_training.py
Production-Ready Mac mini (Apple Silicon M1/M2/M3) Training Pipeline for ZebraID.

Runs the full experimental suite required for the research paper:
  1. baseline_a  (Pop A → Pop A): Within-population baseline (GZGC plains zebra).
  2. baseline_x  (Pop A → Pop B): Cross-population generalization gap (Grevy's zebra).
  3. zebraid     (Mixed A+B):     ⭐ ZebraID novel cross-population model.
  4. resnet50    (Mixed A+B):     ResNet50 backbone ablation.

Optimized for Mac mini M2 Pro:
  - Apple Metal Performance Shaders (MPS) acceleration.
  - Automatic system sleep prevention (`caffeinate`).
  - Optimized micro-batching (batch_size=8, accum_steps=2 -> effective batch=16).
  - Parallel data loading with 4 CPU workers.
  - Automatic evaluation (Rank-1, Rank-5, mAP) and CSV/JSON output generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force unbuffered real-time stdout printing in zsh terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
from zebraid.models.train import train


def prevent_sleep():
    """Attempt to prevent system sleep using macOS caffeinate in a non-blocking process."""
    if sys.platform == "darwin":
        try:
            pid = os.getpid()
            subprocess.Popen(["caffeinate", "-i", "-w", str(pid)])
            print("☕ System sleep prevention active (caffeinate enabled).")
        except Exception as e:
            print(f"⚠️ Could not launch caffeinate: {e}")


def print_system_info():
    """Print hardware and environment diagnostics."""
    print("=" * 65)
    print("  🦓 ZebraID — Mac mini M2 Pro Production Training Pipeline")
    print("=" * 65)
    print(f"  PyTorch Version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("  Hardware Accelerator: Apple Metal Performance Shaders (MPS) ✅")
    elif torch.cuda.is_available():
        print(f"  Hardware Accelerator: CUDA ({torch.cuda.get_device_name(0)}) ✅")
    else:
        print("  Hardware Accelerator: CPU ⚠️ (MPS not available)")
    
    print(f"  Repo Root: {REPO_ROOT}")
    print("=" * 65)


def run_full_pipeline(
    config_path: str = "configs/default.yaml",
    seeds: list[int] = [42],
    num_epochs: int = 30,
    fast_mode: bool = False,
    backbone_override: str | None = None,
    mode_override: str | None = None,
):
    """Execute all training runs and compile comparison results."""
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    total_start_time = time.time()
    
    # Experiments matrix
    if fast_mode or backbone_override == "resnet50":
        print("⚡ Fast mode active: Using lightweight ResNet-50 backbone (224x224, 8x faster).")
        experiments = [
            {"mode": "baseline_a", "backbone": "resnet50", "desc": "Fast Baseline A (ResNet-50)"},
            {"mode": "zebraid",    "backbone": "resnet50", "desc": "Fast ZebraID (ResNet-50)"},
        ]
    else:
        experiments = [
            {"mode": "baseline_a", "backbone": "megadescriptor", "desc": "Baseline A (Pop A -> Pop A)"},
            {"mode": "baseline_x", "backbone": "megadescriptor", "desc": "Baseline X (Pop A -> Pop B Generalization Gap)"},
            {"mode": "zebraid",    "backbone": "megadescriptor", "desc": "ZebraID (Mixed A+B -> Both Populations)"},
            {"mode": "zebraid",    "backbone": "resnet50",       "desc": "ResNet-50 Ablation (Mixed A+B)"},
        ]

    if mode_override:
        experiments = [e for e in experiments if e["mode"] == mode_override]
        if not experiments:
            print(f"⚠️ No experiments matched mode '{mode_override}' with current config.")
            return

    for seed in seeds:
        print(f"\n🌱 Executing Training Seed: {seed}")
        print("-" * 65)

        for exp in experiments:
            mode = exp["mode"]
            backbone = exp["backbone"]
            desc = exp["desc"]

            print(f"\n🚀 Running Experiment: {desc}")
            print(f"   Config: mode={mode}, backbone={backbone}, seed={seed}, epochs={num_epochs}")
            print("-" * 65)

            t0 = time.time()
            res = train(
                config_path=config_path,
                mode=mode,
                backbone_name=backbone,
                split_seed=seed,
                num_epochs=num_epochs,
            )
            elapsed = time.time() - t0

            row = {
                "experiment": desc,
                "mode": mode,
                "backbone": backbone,
                "seed": seed,
                "rank1_pop_a": round(res["rank1_a"], 4),
                "map_pop_a": round(res["map_a"], 4),
                "rank1_pop_b": round(res["rank1_b"], 4) if not (isinstance(res["rank1_b"], float) and res["rank1_b"] != res["rank1_b"]) else "N/A",
                "map_pop_b": round(res["map_b"], 4) if not (isinstance(res["map_b"], float) and res["map_b"] != res["map_b"]) else "N/A",
                "checkpoint": res["checkpoint_path"],
                "duration_seconds": round(elapsed, 1),
            }
            summary_rows.append(row)

            print(f"✅ Finished {mode} ({backbone}) in {elapsed/60:.1f} mins.")
            print(f"   Pop A: Rank-1 = {res['rank1_a']:.4f}, mAP = {res['map_a']:.4f}")
            if mode in ("baseline_x", "zebraid"):
                print(f"   Pop B: Rank-1 = {res['rank1_b']:.4f}, mAP = {res['map_b']:.4f}")

    # Export full summary table
    csv_path = results_dir / "training_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = results_dir / "final_paper_results.json"
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 65)
    print("🎉 ALL TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total Execution Time: {total_elapsed/3600:.2f} hours")
    print(f"📊 Results Saved To:")
    print(f"   • {csv_path}")
    print(f"   • {json_path}")
    print("=" * 65)
    
    # ── Paper Tables Generation ──
    print("\n📝 Generating Statistical Paper Tables...")
    try:
        subprocess.run([sys.executable, "scripts/generate_paper_tables.py"], check=True)
    except Exception as e:
        print(f"⚠️ Failed to generate paper tables: {e}")


def main():
    parser = argparse.ArgumentParser(description="ZebraID Mac Production Training Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds for statistical validation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--fast", action="store_true", help="Use lightweight ResNet-50 backbone for fast Mac mini testing")
    parser.add_argument("--backbone", type=str, default=None, choices=["megadescriptor", "resnet50"], help="Backbone model choice")
    parser.add_argument("--mode", type=str, default=None, choices=["baseline_a", "baseline_x", "zebraid"], help="Run only a specific experiment mode")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    prevent_sleep()
    print_system_info()
    
    run_full_pipeline(
        config_path=args.config,
        seeds=args.seeds,
        num_epochs=args.epochs,
        fast_mode=args.fast,
        backbone_override=args.backbone,
        mode_override=args.mode,
    )


if __name__ == "__main__":
    main()
