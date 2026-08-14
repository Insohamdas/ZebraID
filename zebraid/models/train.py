"""
zebraid/models/train.py
Training loop for ZebraID embedding models.

Supports three training configurations (for the paper's comparison table):
  - 'baseline_a':  Train on population A only, evaluate on A.
  - 'baseline_x':  Train on population A only, evaluate on population B.
                   Quantifies the cross-population generalization gap.
  - 'zebraid':     Train on MIXED A+B batches, evaluate on held-out
                   individuals from both populations.

Usage:
    # On Google Colab T4 GPU:
    from zebraid.models.train import train
    train(config_path="configs/default.yaml", mode="zebraid")

    # On Mac M2/M3 (MPS — Apple Metal):
    train(config_path="configs/default.yaml", mode="zebraid")
    # Device is auto-detected: cuda > mps > cpu

Memory per device:
    CUDA T4  (15 GB): batch_size=4, accum_steps=4 → effective 16
    Mac MPS  (16 GB): batch_size=8, accum_steps=2 → effective 16
    CPU only         : batch_size=2, accum_steps=8 → effective 16 (very slow)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import numpy as np
import random
import time
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import yaml

from zebraid.data.dataset import ZebraDataset, CombinedZebraDataset, POP_A, POP_B
from zebraid.data.transforms import train_transforms, eval_transforms
from zebraid.data.mixed_batch_sampler import MixedPopulationBatchSampler
from zebraid.models.backbone import build_embedder
from zebraid.models.loss import TripletLossWithMining
from zebraid.models.evaluate import compute_cmc_map
from zebraid.reporting.experiment_tracker import save_experiment_info
from zebraid.reporting.research_report import generate_research_report


TrainingMode = Literal["baseline_a", "baseline_x", "zebraid"]

def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ── Device helpers ────────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    """
    Auto-select the best available device:
      1. CUDA  (Nvidia GPU — Colab T4/A100)
      2. MPS   (Apple Silicon — Mac M1/M2/M3)
      3. CPU   (fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_config(device: torch.device) -> tuple[bool, str]:
    """
    Returns (use_amp, autocast_device_type) for the given device.
    - CUDA  → AMP with GradScaler enabled (FP16)
    - MPS   → autocast enabled (BF16), GradScaler disabled (not supported on MPS)
    - CPU   → no AMP
    """
    if device.type == "cuda":
        return True, "cuda"
    if device.type == "mps":
        # MPS supports native autocast in PyTorch
        return True, "mps"
    return False, "cpu"


def _free_gpu_memory(device: torch.device) -> None:
    """Release cached GPU memory before allocating a new model."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _try_init_wandb(cfg: dict, run_name: str) -> Optional[object]:
    if not cfg["training"].get("use_wandb", False):
        return None
    try:
        import wandb
        run = wandb.init(
            project=cfg["training"]["wandb_project"],
            name=run_name,
            config=cfg,
        )
        return run
    except Exception as e:
        print(f"[wandb] Could not initialize W&B ({e}). Falling back to CSV logging.")
        return None


# ── Main training function ────────────────────────────────────────────────────

def train(
    config_path: str = "configs/default.yaml",
    mode: TrainingMode = "zebraid",
    split_seed: int = 42,
    backbone_name: str = "megadescriptor",
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Train a ZebraEmbedder and return evaluation metrics.

    Device is auto-detected: CUDA > MPS (Apple Silicon) > CPU.
    Gradient accumulation is used to keep per-step memory low.
    """
    # Support argument aliases
    if "training_mode" in kwargs:
        mode = kwargs["training_mode"]
    if "seed" in kwargs:
        split_seed = kwargs["seed"]
    if "epochs" in kwargs and num_epochs is None:
        num_epochs = kwargs["epochs"]

    seed_everything(split_seed)

    # ── Set env var to reduce CUDA memory fragmentation ──────────────────────
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg    = _load_config(config_path)
    device = _best_device()

    # ── Per-device batch defaults ────────────────────────────────────────────
    # Config drives batch_size and accum_steps; we override for MPS/CPU if
    # the config still has the CUDA-tuned values (to avoid OOM on first run).
    batch_size  = cfg["training"].get("batch_size",  4)
    accum_steps = cfg["training"].get("accum_steps", 4)

    # Auto-adjust for MPS (Apple Silicon) — 16 GB unified memory can handle more
    if device.type == "mps" and batch_size <= 4:
        batch_size  = 8   # MPS is bandwidth-efficient; 8 fits easily in 16 GB
        accum_steps = 2   # still gives effective batch = 16

    effective_batch = batch_size * accum_steps
    if num_epochs is None:
        num_epochs = cfg["training"]["num_epochs"]

    print(
        f"[train] device={device} ({device.type.upper()}), mode={mode}, "
        f"backbone={backbone_name}, seed={split_seed}\n"
        f"        batch_size={batch_size}, accum_steps={accum_steps}, "
        f"effective_batch={effective_batch}, epochs={num_epochs}"
    )

    out_dir = (
        Path(output_dir or cfg["paths"]["checkpoint_dir"])
        / mode / backbone_name / f"seed{split_seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_experiment_info(
        out_dir=out_dir, 
        cfg=cfg, 
        command_args=sys.argv, 
        seed=split_seed, 
        mode=mode, 
        backbone=backbone_name
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    from zebraid.data.loaders import build_datasets
    img_size = cfg["model"].get("img_size", 384 if backbone_name == "megadescriptor" else 224)
    use_flip = cfg.get("data", {}).get("use_horizontal_flip", False)
    min_images = cfg.get("data", {}).get("min_images_per_individual", 2)
    t_train  = train_transforms(img_size, use_horizontal_flip=use_flip)
    t_eval   = eval_transforms(img_size)

    ds_a_train, ds_b_train = build_datasets(
        "train", transform=t_train, split_seed=split_seed, min_images_per_individual=min_images
    )
    ds_a_val,   ds_b_val   = build_datasets(
        "val",   transform=t_eval,  split_seed=split_seed, min_images_per_individual=min_images
    )
    ds_a_test,  ds_b_test  = build_datasets(
        "test",  transform=t_eval,  split_seed=split_seed, min_images_per_individual=min_images
    )

    # ── Rigorous Identity Leakage Assertions ─────────────────────────────────
    a_tr, a_va, a_te = set(ds_a_train.individual_ids), set(ds_a_val.individual_ids), set(ds_a_test.individual_ids)
    b_tr, b_va, b_te = set(ds_b_train.individual_ids), set(ds_b_val.individual_ids), set(ds_b_test.individual_ids)

    assert a_tr.isdisjoint(a_va), f"FATAL: Pop A Train/Val identity leak detected ({len(a_tr & a_va)} overlapping IDs)!"
    assert a_tr.isdisjoint(a_te), f"FATAL: Pop A Train/Test identity leak detected ({len(a_tr & a_te)} overlapping IDs)!"
    assert a_va.isdisjoint(a_te), f"FATAL: Pop A Val/Test identity leak detected ({len(a_va & a_te)} overlapping IDs)!"

    assert b_tr.isdisjoint(b_va), f"FATAL: Pop B Train/Val identity leak detected ({len(b_tr & b_va)} overlapping IDs)!"
    assert b_tr.isdisjoint(b_te), f"FATAL: Pop B Train/Test identity leak detected ({len(b_tr & b_te)} overlapping IDs)!"
    assert b_va.isdisjoint(b_te), f"FATAL: Pop B Val/Test identity leak detected ({len(b_va & b_te)} overlapping IDs)!"

    all_a_ids = a_tr | a_va | a_te
    all_b_ids = b_tr | b_va | b_te
    assert all_a_ids.isdisjoint(all_b_ids), f"FATAL: Pop A and Pop B ID collision detected ({len(all_a_ids & all_b_ids)} overlapping IDs)!"

    print("\n" + "="*60)
    print("📊 DATASET SPLIT & LEAKAGE DIAGNOSTICS")
    print("="*60)
    print(f"Population A (GZGC Plains Zebra, min_images>={min_images}):")
    print(f"  Total Eligible Identities: {len(all_a_ids)}")
    print(f"  Train: {len(ds_a_train):>5d} images across {len(a_tr):>4d} identities")
    print(f"  Val:   {len(ds_a_val):>5d} images across {len(a_va):>4d} identities")
    print(f"  Test:  {len(ds_a_test):>5d} images across {len(a_te):>4d} identities")
    print(f"Population B (Mpala Grevy's Zebra, min_images>={min_images}):")
    print(f"  Total Discovered: {getattr(ds_b_train, 'total_individuals', len(all_b_ids))} ({getattr(ds_b_train, 'excluded_singletons', 0)} singletons excluded)")
    print(f"  Total Eligible Identities: {len(all_b_ids)}")
    print(f"  Train: {len(ds_b_train):>5d} images across {len(b_tr):>4d} identities")
    print(f"  Val:   {len(ds_b_val):>5d} images across {len(b_va):>4d} identities")
    print(f"  Test:  {len(ds_b_test):>5d} images across {len(b_te):>4d} identities")
    print("Identity Overlap Checks:")
    print(f"  Pop A: Train ∩ Val = {len(a_tr & a_va)} | Train ∩ Test = {len(a_tr & a_te)} | Val ∩ Test = {len(a_va & a_te)} (ZERO LEAKAGE ✅)")
    print(f"  Pop B: Train ∩ Val = {len(b_tr & b_va)} | Train ∩ Test = {len(b_tr & b_te)} | Val ∩ Test = {len(b_va & b_te)} (ZERO LEAKAGE ✅)")
    print(f"  Cross: Pop A ∩ Pop B = {len(all_a_ids & all_b_ids)} (ZERO COLLISION ✅)")
    print("="*60 + "\n")

    # ── Build dataloaders ────────────────────────────────────────────────────
    # pin_memory only helps with CUDA; disable for MPS and CPU
    pin = device.type == "cuda"
    # Use 2 workers to keep the GPU fed with data
    num_workers = 2

    if mode in ("baseline_a", "baseline_x"):
        train_loader = DataLoader(
            ds_a_train,
            batch_size=batch_size,
            sampler=RandomSampler(ds_a_train),
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn,
        )
    else:  # zebraid — mixed batch
        combined_train = CombinedZebraDataset(ds_a_train, ds_b_train)
        mixed_sampler  = MixedPopulationBatchSampler(
            combined_train,
            batch_size=batch_size,
            ratio_a=cfg["training"]["mixed_batch_ratio"],
            seed=split_seed,
        )
        train_loader = DataLoader(
            combined_train,
            batch_sampler=mixed_sampler,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn,
        )

    # ── Model + optimizer ────────────────────────────────────────────────────
    _free_gpu_memory(device)
    model = build_embedder(
        backbone_name=backbone_name,
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=True,
        device=device,
    )

    # Linear Probing: Freeze backbone to preserve pretrained features
    base_lr = cfg["training"]["learning_rate"]
    backbone_params = []
    projector_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            backbone_params.append(param)
        else:
            projector_params.append(param)

    optimizer = torch.optim.AdamW(
        projector_params,
        lr=base_lr,
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Warmup + Cosine Annealing Scheduler
    try:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = min(3, max(1, num_epochs // 5))  # Usually 1-3 epochs
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    except ImportError:
        # Fallback for older PyTorch versions (< 1.10)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = TripletLossWithMining(
        margin=cfg["training"]["triplet_margin"],
        mining_type=cfg["training"]["triplet_mining"],
    )

    # ── AMP setup ────────────────────────────────────────────────────────────
    use_amp, amp_device_type = _amp_config(device)
    # GradScaler only works on CUDA (not MPS)
    use_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # ── Logging ───────────────────────────────────────────────────────────────
    run_name     = f"{mode}_{backbone_name}_seed{split_seed}"
    wandb_run    = _try_init_wandb(cfg, run_name)
    csv_log_path = out_dir / "training_log.csv"
    csv_fields   = [
        "epoch", "train_loss", 
        "rank1_a", "rank1_b", "rank1_b_multi",
        "map_a", "map_b", "map_b_multi",
        "lr_backbone", "lr_projector", 
        "throughput_img_sec", "dataloader_wait_pct", "peak_memory_mb"
    ]

    with open(csv_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Startup Summary ───────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    backbone_params_count = sum(p.numel() for p in backbone_params)
    projector_params_count = sum(p.numel() for p in projector_params)
    
    print("\n" + "="*60)
    print("🚀 ZEBRAID TRAINING STARTUP SUMMARY")
    print("="*60)
    print(f"Model:                 {backbone_name}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {frozen_params:,}")
    print(f"Backbone parameters:   {backbone_params_count:,}")
    print(f"Projector parameters:  {projector_params_count:,}")
    print(f"pretrained=True used:  True")
    
    input_size = (
        cfg.get("model", {}).get("input_size")
        or cfg.get("data", {}).get("image_size")
        or "Unknown"
    )
    print(f"Input image size:      {input_size}")
    
    print("-" * 60)
    print("Optimizer Groups:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr = {param_group['lr']}")
    print("-" * 60)
    
    _warmup = warmup_epochs if 'warmup_epochs' in locals() else 0
    print(f"Warmup schedule:       {_warmup} epochs (LinearLR -> CosineAnnealingLR)")
    
    batch_sz = cfg['training']['batch_size']
    accum = cfg['training']['accum_steps']
    print(f"Batch size:            {batch_sz} (accum: {accum}) -> Effective: {batch_sz * accum}")
    
    if mode == "zebraid":
        k_instances = 2
        p_a = max(1, int(batch_sz * 0.5) // k_instances)
        p_b = max(1, (batch_sz - int(batch_sz * 0.5)) // k_instances)
        print(f"Batch composition:     {p_a} PopA indivs, {p_b} PopB indivs (K={k_instances} images/indiv)")
    print("="*60 + "\n")

    # ── Training loop with gradient accumulation ──────────────────────────────
    best_rank1 = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        # Protect backbone BN statistics from being ruined by micro-batches
        model.backbone.eval()

        # Update sampler epoch for deterministic epoch-aware shuffle
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        t0 = time.time()
        
        # Profiling accumulators
        data_wait_time = 0.0
        compute_time = 0.0
        num_images_processed = 0
        
        optimizer.zero_grad()
        
        # Reset memory tracking if CUDA
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        t_data_start = time.time()

        for step, (images, individual_ids, _) in enumerate(train_loader):
            data_wait_time += time.time() - t_data_start
            t_comp_start = time.time()
            
            images         = images.to(device, non_blocking=True)
            individual_ids = individual_ids.to(device, non_blocking=True)
            batch_size     = images.size(0)
            num_images_processed += batch_size

            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                embeddings = model(images)
                loss       = criterion(embeddings, individual_ids)
                loss       = loss / accum_steps  # scale for accumulation

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            compute_time += time.time() - t_comp_start

            total_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader) or step == 0:
                print(
                    f"  [Epoch {epoch:02d}/{num_epochs:02d}] Step [{step+1:03d}/{len(train_loader):03d}] "
                    f"Loss: {loss.item() * accum_steps:.4f}",
                    flush=True,
                )
            t_data_start = time.time()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        # ── Evaluation ───────────────────────────────────────────────────────
        model.eval()
        _free_gpu_memory(device)
        with torch.no_grad():
            metrics_a = compute_cmc_map(model, ds_a_val, device, top_k=5)
            if mode in ("baseline_x", "zebraid"):
                metrics_b = compute_cmc_map(model, ds_b_val, device, top_k=5)
            else:
                metrics_b = {
                    "rank1": float("nan"), "map": float("nan"),
                    "rank1_multi": float("nan"), "map_multi": float("nan"),
                    "n_queries": 0, "n_queries_multi": 0, "n_singletons": 0
                }

        lrs = scheduler.get_last_lr()
        lr_bb = 0.0  # backbone is frozen
        lr_proj = lrs[0]

        def _safe_round(v: float) -> float:
            return round(v, 4) if v == v else float("nan")  # nan check

        # ── Profiling Metrics ────────────────────────────────────────────────
        throughput = num_images_processed / elapsed if elapsed > 0 else 0
        total_train_time = data_wait_time + compute_time
        dl_wait_pct = (data_wait_time / total_train_time * 100.0) if total_train_time > 0 else 0.0
        
        peak_mem_mb = 0.0
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif device.type == "mps":
            peak_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

        row = {
            "epoch":              epoch,
            "train_loss":         round(avg_loss, 4),
            "rank1_a":            _safe_round(metrics_a["rank1"]),
            "rank1_b":            _safe_round(metrics_b["rank1"]),
            "rank1_b_multi":      _safe_round(metrics_b.get("rank1_multi", float("nan"))),
            "map_a":              _safe_round(metrics_a["map"]),
            "map_b":              _safe_round(metrics_b["map"]),
            "map_b_multi":        _safe_round(metrics_b.get("map_multi", float("nan"))),
            "lr_backbone":        lr_bb,
            "lr_projector":       lr_proj,
            "throughput_img_sec": round(throughput, 1),
            "dataloader_wait_pct": round(dl_wait_pct, 1),
            "peak_memory_mb":     round(peak_mem_mb, 1),
        }

        print(
            f"[Epoch {epoch:03d}/{num_epochs}] loss={avg_loss:.4f} "
            f"R1_A={metrics_a['rank1']:.3f} R1_B={metrics_b['rank1']:.3f} "
            f"(R1_B_Multi={metrics_b.get('rank1_multi', float('nan')):.3f}) "
            f"mAP_A={metrics_a['map']:.3f} mAP_B={metrics_b['map']:.3f} "
            f"| LR(proj)={lr_proj:.2e} "
            f"({elapsed:.1f}s | {throughput:.1f} im/s | Mem: {peak_mem_mb:.0f} MB | DL Wait: {dl_wait_pct:.1f}%)"
        )

        with open(csv_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if wandb_run:
            wandb_run.log(row)

        # Save best checkpoint (rich state dictionary with backward-compatible loading)
        rank1_a = metrics_a["rank1"]
        rank1_b = metrics_b["rank1"] if metrics_b["rank1"] == metrics_b["rank1"] else 0.0
        val_score = (rank1_a + rank1_b) / 2
        if val_score > best_rank1:
            best_rank1 = val_score
            checkpoint_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_metrics": {
                    "rank1_a": metrics_a["rank1"],
                    "rank1_b": metrics_b["rank1"],
                    "rank1_b_multi": metrics_b.get("rank1_multi", float("nan")),
                    "map_a": metrics_a["map"],
                    "map_b": metrics_b["map"],
                    "map_b_multi": metrics_b.get("map_multi", float("nan")),
                    "val_score": val_score,
                },
                "rng_state": torch.get_rng_state(),
                "config": cfg,
            }
            torch.save(checkpoint_payload, out_dir / "best_model.pt")

    # ── Final metrics ─────────────────────────────────────────────────────────
    _free_gpu_memory(device)
    raw_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        model.load_state_dict(raw_ckpt["model"])
    else:
        model.load_state_dict(raw_ckpt)

    model.eval()
    with torch.no_grad():
        final_a = compute_cmc_map(
            model, ds_a_val, device, top_k=5, 
            save_examples_dir=out_dir / "retrieval_examples" / "pop_a"
        )
        final_b = (
            compute_cmc_map(
                model, ds_b_val, device, top_k=5,
                save_examples_dir=out_dir / "retrieval_examples" / "pop_b"
            )
            if mode in ("baseline_x", "zebraid")
            else {
                "rank1": float("nan"), "map": float("nan"),
                "rank1_multi": float("nan"), "map_multi": float("nan"),
                "n_queries": 0, "n_queries_multi": 0, "n_singletons": 0
            }
        )

    result = {
        "mode":              mode,
        "backbone":          backbone_name,
        "seed":              split_seed,
        "rank1_a":           final_a["rank1"],
        "rank1_b":           final_b["rank1"],
        "rank1_b_multi":     final_b.get("rank1_multi", float("nan")),
        "map_a":             final_a["map"],
        "map_b":             final_b["map"],
        "map_b_multi":       final_b.get("map_multi", float("nan")),
        "n_queries_a":       final_a.get("n_queries", len(ds_a_val)),
        "n_queries_b":       final_b.get("n_queries", len(ds_b_val) if mode in ("baseline_x", "zebraid") else 0),
        "n_queries_b_multi": final_b.get("n_queries_multi", 0),
        "n_singletons_b":    final_b.get("n_singletons", 0),
        "checkpoint_path":   str(out_dir / "best_model.pt"),
    }

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[train] Done. Metrics: {out_dir / 'final_metrics.json'}")
    
    # Auto-generate research report and plots
    generate_research_report(out_dir)
    
    if wandb_run:
        wandb_run.finish()

    return result