"""
scripts/final_validation.py
Pre-flight checklist to ensure CVPR-ready settings before running 
the final 30-epoch ablation studies.
"""

import sys
import re
from pathlib import Path
import yaml

def fatal_error(msg):
    print(f"❌ FATAL: {msg}")
    print("Cannot proceed with 30-epoch run. Please fix the codebase.")
    sys.exit(1)

def run_preflight_checks():
    print("=== Final Validation Pre-Flight Checklist ===")
    
    # 1. Load config
    cfg_path = Path("configs/default.yaml")
    if not cfg_path.exists():
        fatal_error(f"Config not found at {cfg_path}")
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    use_flip = cfg.get("data", {}).get("use_horizontal_flip", None)
    if use_flip is True:
        fatal_error("RandomHorizontalFlip is ENABLED in default.yaml. This breaks biometric uniqueness.")
    else:
        print("✅ RandomHorizontalFlip is disabled or missing (safe).")
        
    num_epochs = cfg.get("training", {}).get("num_epochs", 0)
    if num_epochs < 30:
        print(f"⚠️ Warning: num_epochs={num_epochs} in default.yaml. Are you sure you're doing the final run?")
    else:
        print("✅ num_epochs >= 30.")

    # 2. Check train.py for pretrained=True
    train_path = Path("zebraid/models/train.py")
    if not train_path.exists():
        fatal_error("train.py not found.")
        
    with open(train_path, "r") as f:
        train_code = f.read()
        
    if "pretrained=True" not in train_code:
        fatal_error("pretrained=True not found in train.py. Backbone is initializing from scratch!")
    else:
        print("✅ pretrained=True found.")
        
    # 3. Check for seed_everything
    if "seed_everything" not in train_code:
        fatal_error("seed_everything not found in train.py. Runs will not be reproducible.")
    else:
        print("✅ Global stochastic determinism enabled.")
        
    # 4. Check for differential learning rates
    if "backbone_params" not in train_code or "projector_params" not in train_code:
        fatal_error("Differential learning rates not found in train.py. Backbone will collapse.")
    else:
        print("✅ Differential learning rates implemented.")
        
    # 5. Check for worker_init_fn
    if "worker_init_fn" not in train_code:
        fatal_error("worker_init_fn not passed to DataLoader. Background workers lack deterministic seeds.")
    else:
        print("✅ worker_init_fn protects dataloader determinism.")

    # 6. Check for experiment tracking
    if "save_experiment_info" not in train_code:
        fatal_error("Experiment tracking missing. Metrics will not be saved properly.")
    else:
        print("✅ Experiment Tracker is wired in.")
        
    print("\n🚀 Pre-flight checks passed. Cleared for 30-epoch training.")

if __name__ == "__main__":
    run_preflight_checks()
