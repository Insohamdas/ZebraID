"""
zebraid/reporting/research_report.py
Generates CVPR-ready evaluation reports and plots from experiment logs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def generate_research_report(checkpoint_dir: str | Path):
    """
    Parses training_log.csv and experiment_info.json in the checkpoint_dir
    to generate matplotlib plots and a comprehensive markdown report.
    """
    ckpt_dir = Path(checkpoint_dir)
    log_csv = ckpt_dir / "training_log.csv"
    info_json = ckpt_dir / "experiment_info.json"
    
    if not log_csv.exists() or not info_json.exists():
        print(f"⚠️ Cannot generate report: missing logs in {ckpt_dir}")
        return
        
    with open(info_json, "r") as f:
        info = json.load(f)
        
    epochs = []
    losses = []
    r1_a = []
    r1_b = []
    map_a = []
    map_b = []
    lr_bb = []
    lr_proj = []
    
    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            r1_a.append(float(row["rank1_a"]) if row["rank1_a"] != "nan" else 0.0)
            r1_b.append(float(row["rank1_b"]) if row["rank1_b"] != "nan" else 0.0)
            map_a.append(float(row["map_a"]) if row["map_a"] != "nan" else 0.0)
            map_b.append(float(row["map_b"]) if row["map_b"] != "nan" else 0.0)
            lr_bb.append(float(row.get("lr_backbone", 0.0)))
            lr_proj.append(float(row.get("lr_projector", 0.0)))
            
    if not epochs:
        return
        
    plots_dir = ckpt_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if plt is not None:
        # Plot Loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker='o', label="Train Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / "loss_curve.png")
        plt.close()
        
        # Plot Rank-1
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, r1_a, marker='s', label="Rank-1 (Pop A)")
        if any(v > 0 for v in r1_b):
            plt.plot(epochs, r1_b, marker='^', label="Rank-1 (Pop B)")
        plt.title("Rank-1 Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Rank-1 (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / "rank1_curve.png")
        plt.close()
        
        # Plot mAP
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, map_a, marker='s', label="mAP (Pop A)")
        if any(v > 0 for v in map_b):
            plt.plot(epochs, map_b, marker='^', label="mAP (Pop B)")
        plt.title("Mean Average Precision (mAP) Curve")
        plt.xlabel("Epoch")
        plt.ylabel("mAP (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / "map_curve.png")
        plt.close()

        # Plot LR
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, lr_bb, marker='o', label="Backbone LR")
        plt.plot(epochs, lr_proj, marker='x', label="Projector LR")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / "lr_curve.png")
        plt.close()

    # Generate Markdown Report
    report_md = ckpt_dir / "research_report.md"
    
    best_idx = max(range(len(epochs)), key=lambda i: r1_a[i] + r1_b[i])
    
    with open(report_md, "w") as f:
        f.write(f"# Research Report: {info['experiment_mode']} ({info['backbone']})\n\n")
        f.write(f"**Date:** {info['timestamp']}\\n")
        f.write(f"**Git Commit:** `{info['git_commit']}`\\n")
        f.write(f"**Hardware:** {info['gpu_model']} | **OS:** {info['os']}\\n")
        f.write(f"**Random Seed:** {info['random_seed']}\\n\n")
        
        f.write("## 1. Final Metrics (Best Epoch)\n")
        f.write(f"- **Best Epoch:** {epochs[best_idx]}\n")
        f.write(f"- **Pop A Rank-1:** {r1_a[best_idx]:.4f}\n")
        f.write(f"- **Pop A mAP:** {map_a[best_idx]:.4f}\n")
        if r1_b[best_idx] > 0:
            f.write(f"- **Pop B Rank-1:** {r1_b[best_idx]:.4f}\n")
            f.write(f"- **Pop B mAP:** {map_b[best_idx]:.4f}\n")
            
        f.write("\n## 2. Training Dynamics\n")
        f.write("![Loss Curve](plots/loss_curve.png)\n\n")
        f.write("![Rank-1 Curve](plots/rank1_curve.png)\n\n")
        f.write("![mAP Curve](plots/map_curve.png)\n\n")
        f.write("![LR Curve](plots/lr_curve.png)\n\n")
        
        f.write("\n## 3. Retrieval Examples\n")
        f.write("See `retrieval_examples/` directory for visual breakdown of hard positives and hard failures.\n")
        
    print(f"📊 Auto-generated research report at {report_md}")
