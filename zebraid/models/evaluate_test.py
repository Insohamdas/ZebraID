"""
zebraid/models/evaluate_test.py
Dedicated Held-Out Test Split Evaluator for ZebraID models.

Guarantees:
  1. The TEST split remains completely untouched during model selection.
  2. Evaluates strictly on held-out test identities (split='test').
  3. No model weights or hyperparameters are updated.
  4. Train/Val/Test identity leakage is asserted with zero tolerance.
  5. Outputs distinct test_metrics.json, test_report.md, test_retrieval_examples/, and test_plots/.
  6. Preserves existing best_model.pt, final_metrics.json, and training_log.csv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from zebraid.data.loaders import build_datasets
from zebraid.data.transforms import eval_transforms
from zebraid.models.backbone import build_embedder, ZebraEmbedder
from zebraid.models.evaluate import compute_cmc_map


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_checkpoint(
    checkpoint_path: str | Path,
    backbone_name: str = "megadescriptor",
    embedding_dim: int = 512,
    device: Optional[torch.device] = None,
) -> ZebraEmbedder:
    """Load a trained ZebraEmbedder from a checkpoint file (safe against rich dict format)."""
    if device is None:
        device = _best_device()

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    model = build_embedder(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=False,
        device=device,
    )

    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        model.load_state_dict(raw_ckpt["model"])
    else:
        model.load_state_dict(raw_ckpt)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def evaluate_held_out_test(
    checkpoint_path: str | Path,
    config_path: str = "configs/default.yaml",
    split_seed: int = 42,
    min_images_per_individual: int = 2,
    backbone_name: str = "megadescriptor",
    output_dir: Optional[str | Path] = None,
    save_examples: bool = True,
    save_plots: bool = True,
    device: Optional[torch.device] = None,
    top_k: int = 10,
) -> dict:
    """
    Perform held-out test evaluation on a trained model checkpoint.

    Args:
        checkpoint_path: Path to best_model.pt.
        config_path: Path to configs/default.yaml.
        split_seed: Random seed used for the dataset splits (fixed at 42).
        min_images_per_individual: Minimum images required per individual (default 2).
        backbone_name: 'megadescriptor' or 'resnet50'.
        output_dir: Output directory for test metrics, reports, and examples.
                    Defaults to the checkpoint parent directory.
        save_examples: Whether to generate visual retrieval example images.
        save_plots: Whether to generate CMC and metric plots.
        device: Device to run evaluation on.
        top_k: Top-k ranking evaluation limit (default 10).

    Returns:
        Structured test metrics dictionary.
    """
    if device is None:
        device = _best_device()

    ckpt_path = Path(checkpoint_path).resolve()
    if output_dir is None:
        out_dir = ckpt_path.parent
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    embedding_dim = cfg["model"].get("embedding_dim", 512)
    img_size = cfg["model"].get("img_size", 384 if backbone_name == "megadescriptor" else 224)

    # ── Load model in strict eval mode ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🔬 ZEBRAID HELD-OUT TEST EVALUATION")
    print(f"{'='*70}")
    print(f"Checkpoint:   {ckpt_path}")
    print(f"Backbone:     {backbone_name}")
    print(f"Device:       {device}")
    print(f"Split Seed:   {split_seed}")
    print(f"Output Dir:   {out_dir}")
    print(f"{'='*70}\n")

    model = load_model_checkpoint(
        checkpoint_path=ckpt_path,
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        device=device,
    )

    # ── Build TEST datasets ───────────────────────────────────────────────────
    t_eval = eval_transforms(img_size)
    ds_a_test, ds_b_test = build_datasets(
        "test",
        transform=t_eval,
        split_seed=split_seed,
        min_images_per_individual=min_images_per_individual,
    )

    # Build train and val datasets solely for strict zero-leakage assertions
    ds_a_train, ds_b_train = build_datasets(
        "train",
        transform=None,
        split_seed=split_seed,
        min_images_per_individual=min_images_per_individual,
    )
    ds_a_val, ds_b_val = build_datasets(
        "val",
        transform=None,
        split_seed=split_seed,
        min_images_per_individual=min_images_per_individual,
    )

    # ── Strict Zero-Leakage Identity Assertions ───────────────────────────────
    a_tr = set(ds_a_train.individual_ids)
    a_va = set(ds_a_val.individual_ids)
    a_te = set(ds_a_test.individual_ids)

    b_tr = set(ds_b_train.individual_ids)
    b_va = set(ds_b_val.individual_ids)
    b_te = set(ds_b_test.individual_ids)

    assert a_tr.isdisjoint(a_te), f"FATAL: Pop A Train/Test identity leak! Overlap: {a_tr & a_te}"
    assert a_va.isdisjoint(a_te), f"FATAL: Pop A Val/Test identity leak! Overlap: {a_va & a_te}"
    assert b_tr.isdisjoint(b_te), f"FATAL: Pop B Train/Test identity leak! Overlap: {b_tr & b_te}"
    assert b_va.isdisjoint(b_te), f"FATAL: Pop B Val/Test identity leak! Overlap: {b_va & b_te}"

    all_a = a_tr | a_va | a_te
    all_b = b_tr | b_va | b_te
    assert all_a.isdisjoint(all_b), f"FATAL: Pop A and Pop B ID collision! Overlap: {all_a & all_b}"

    print("✅ Zero-Leakage Verification:")
    print(f"   • Pop A: Train ∩ Test = {len(a_tr & a_te)} | Val ∩ Test = {len(a_va & a_te)}")
    print(f"   • Pop B: Train ∩ Test = {len(b_tr & b_te)} | Val ∩ Test = {len(b_va & b_te)}")
    print(f"   • Cross: Pop A ∩ Pop B = {len(all_a & all_b)}")
    print("----------------------------------------------------------------------")
    print(f"Test Split Counts:")
    print(f"   • Pop A (GZGC Plains Zebra):       {len(ds_a_test)} queries across {len(a_te)} identities")
    print(f"   • Pop B (Mpala Grevy's Zebra):     {len(ds_b_test)} queries across {len(b_te)} identities")
    print("----------------------------------------------------------------------\n")

    # ── Evaluate Population A ─────────────────────────────────────────────────
    examples_dir = out_dir / "test_retrieval_examples"
    print("🔍 Evaluating Population A (GZGC Plains Zebra) on held-out test split...")
    with torch.no_grad():
        res_a = compute_cmc_map(
            model=model,
            dataset=ds_a_test,
            device=device,
            top_k=top_k,
            save_examples_dir=examples_dir / "pop_a" if save_examples else None,
        )

    # ── Evaluate Population B ─────────────────────────────────────────────────
    print("🔍 Evaluating Population B (Mpala Grevy's Zebra) on held-out test split...")
    with torch.no_grad():
        res_b = compute_cmc_map(
            model=model,
            dataset=ds_b_test,
            device=device,
            top_k=top_k,
            save_examples_dir=examples_dir / "pop_b" if save_examples else None,
        )

    # ── Compile Metrics ───────────────────────────────────────────────────────
    test_metrics = {
        "evaluation_type": "HELD_OUT_TEST_EVALUATION",
        "checkpoint_path": str(ckpt_path),
        "backbone": backbone_name,
        "split_seed": split_seed,
        "min_images_per_individual": min_images_per_individual,
        "leakage_checks": {
            "pop_a_train_test_overlap": len(a_tr & a_te),
            "pop_a_val_test_overlap": len(a_va & a_te),
            "pop_b_train_test_overlap": len(b_tr & b_te),
            "pop_b_val_test_overlap": len(b_va & b_te),
            "cross_population_overlap": len(all_a & all_b),
            "zero_leakage_verified": True,
        },
        "population_a": {
            "dataset_name": "GZGC (Plains Zebra)",
            "n_identities": ds_a_test.num_individuals,
            "total_queries": res_a["n_queries"],
            "valid_queries": res_a["n_queries_multi"],
            "singleton_queries": res_a["n_singletons"],
            "all_queries": {
                "rank1": res_a["rank1"],
                "rank5": res_a["rank5"],
                "rank10": res_a["rank10"],
                "map": res_a["map"],
            },
            "multi_image_queries": {
                "rank1": res_a["rank1_multi"],
                "rank5": res_a["rank5_multi"],
                "rank10": res_a["rank10_multi"],
                "map": res_a["map_multi"],
            },
        },
        "population_b": {
            "dataset_name": "Labeled Mpala (Grevy's Zebra)",
            "n_identities": ds_b_test.num_individuals,
            "total_queries": res_b["n_queries"],
            "valid_queries": res_b["n_queries_multi"],
            "singleton_queries": res_b["n_singletons"],
            "all_queries": {
                "rank1": res_b["rank1"],
                "rank5": res_b["rank5"],
                "rank10": res_b["rank10"],
                "map": res_b["map"],
            },
            "multi_image_queries": {
                "rank1": res_b["rank1_multi"],
                "rank5": res_b["rank5_multi"],
                "rank10": res_b["rank10_multi"],
                "map": res_b["map_multi"],
            },
        },
    }

    # ── Write test_metrics.json (Never overwrite final_metrics.json) ──────────
    test_metrics_path = out_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"💾 Saved test metrics to {test_metrics_path}")

    # ── Generate Plots ────────────────────────────────────────────────────────
    plots_dir = out_dir / "test_plots"
    if save_plots and plt is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: CMC Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ranks = ["Rank-1", "Rank-5", "Rank-10"]
        pop_a_ranks = [res_a["rank1"] * 100, res_a["rank5"] * 100, res_a["rank10"] * 100]
        pop_b_ranks = [res_b["rank1"] * 100, res_b["rank5"] * 100, res_b["rank10"] * 100]

        x = np.arange(len(ranks))
        width = 0.35

        ax.bar(x - width/2, pop_a_ranks, width, label="Pop A (Plains Zebra)", color="#1f77b4")
        ax.bar(x + width/2, pop_b_ranks, width, label="Pop B (Grevy's Zebra)", color="#ff7f0e")

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"ZebraID Held-Out Test Evaluation — Cumulative Match Characteristic (CMC)")
        ax.set_xticks(x)
        ax.set_xticklabels(ranks)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

        for i in range(len(ranks)):
            ax.text(x[i] - width/2, pop_a_ranks[i] + 1.5, f"{pop_a_ranks[i]:.1f}%", ha='center', fontweight='bold')
            ax.text(x[i] + width/2, pop_b_ranks[i] + 1.5, f"{pop_b_ranks[i]:.1f}%", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / "test_cmc_curve.png", dpi=200)
        plt.close()

        # Plot 2: mAP Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(6, 5))
        maps = [res_a["map"] * 100, res_b["map"] * 100]
        pops = ["Pop A (Plains)", "Pop B (Grevy's)"]
        bars = ax.bar(pops, maps, color=["#1f77b4", "#ff7f0e"], width=0.5)
        ax.set_ylabel("mAP (%)")
        ax.set_title("ZebraID Held-Out Test Evaluation — Mean Average Precision (mAP)")
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / "test_map_comparison.png", dpi=200)
        plt.close()

    # ── Generate Markdown Test Report ─────────────────────────────────────────
    test_report_path = out_dir / "test_report.md"
    with open(test_report_path, "w") as f:
        f.write("# ZebraID Held-Out Test Evaluation Report\n\n")
        f.write(f"**Checkpoint:** `{ckpt_path.name}`  \n")
        f.write(f"**Backbone:** `{backbone_name}`  \n")
        f.write(f"**Evaluation Mode:** Held-Out Unseen Test Split (`split='test'`)  \n")
        f.write(f"**Dataset Split Seed:** `{split_seed}` (Zero Leakage Verified ✅)  \n\n")

        f.write("## 1. Zero Identity Leakage Audit\n\n")
        f.write("| Split Comparison | Population A (Plains) | Population B (Grevy's) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Train ∩ Test Identity Overlap** | {len(a_tr & a_te)} (0 expected) | {len(b_tr & b_te)} (0 expected) |\n")
        f.write(f"| **Val ∩ Test Identity Overlap** | {len(a_va & a_te)} (0 expected) | {len(b_va & b_te)} (0 expected) |\n")
        f.write(f"| **Pop A ∩ Pop B ID Collision** | {len(all_a & all_b)} (0 expected) | {len(all_a & all_b)} (0 expected) |\n\n")

        f.write("## 2. Test Split Demographics & Query Counts\n\n")
        f.write("| Demographic Attribute | Population A (GZGC Plains) | Population B (Mpala Grevy's) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Held-Out Test Identities** | {ds_a_test.num_individuals} | {ds_b_test.num_individuals} |\n")
        f.write(f"| **Total Test Queries ($N$)** | {res_a['n_queries']} | {res_b['n_queries']} |\n")
        f.write(f"| **Valid Queries (Multi-Image)** | {res_a['n_queries_multi']} | {res_b['n_queries_multi']} |\n")
        f.write(f"| **Singleton Queries** | {res_a['n_singletons']} | {res_b['n_singletons']} |\n\n")

        f.write("## 3. Held-Out Test Performance Metrics\n\n")
        f.write("### Population A — Plains Zebra (GZGC)\n")
        f.write("| Metric | All Queries ($N=" + str(res_a['n_queries']) + "$) | Multi-Image Queries ($N=" + str(res_a['n_queries_multi']) + "$) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Rank-1 Accuracy** | **{res_a['rank1']*100:.2f}%** | **{res_a['rank1_multi']*100:.2f}%** |\n")
        f.write(f"| **Rank-5 Accuracy** | **{res_a['rank5']*100:.2f}%** | **{res_a['rank5_multi']*100:.2f}%** |\n")
        f.write(f"| **Rank-10 Accuracy** | **{res_a['rank10']*100:.2f}%** | **{res_a['rank10_multi']*100:.2f}%** |\n")
        f.write(f"| **mAP** | **{res_a['map']*100:.2f}%** | **{res_a['map_multi']*100:.2f}%** |\n\n")

        f.write("### Population B — Grevy's Zebra (Mpala)\n")
        f.write("| Metric | All Queries ($N=" + str(res_b['n_queries']) + "$) | Multi-Image Queries ($N=" + str(res_b['n_queries_multi']) + "$) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Rank-1 Accuracy** | **{res_b['rank1']*100:.2f}%** | **{res_b['rank1_multi']*100:.2f}%** |\n")
        f.write(f"| **Rank-5 Accuracy** | **{res_b['rank5']*100:.2f}%** | **{res_b['rank5_multi']*100:.2f}%** |\n")
        f.write(f"| **Rank-10 Accuracy** | **{res_b['rank10']*100:.2f}%** | **{res_b['rank10_multi']*100:.2f}%** |\n")
        f.write(f"| **mAP** | **{res_b['map']*100:.2f}%** | **{res_b['map_multi']*100:.2f}%** |\n\n")

        if save_plots:
            f.write("## 4. Test Evaluation Plots\n\n")
            f.write("![Test CMC Curve](test_plots/test_cmc_curve.png)\n\n")
            f.write("![Test mAP Comparison](test_plots/test_map_comparison.png)\n\n")

        if save_examples:
            f.write("## 5. Visual Retrieval Examples\n\n")
            f.write("Visual retrieval queries with top-k matching galleries are saved in `test_retrieval_examples/`.\n")

    print(f"📊 Generated test evaluation report: {test_report_path}")
    return test_metrics
