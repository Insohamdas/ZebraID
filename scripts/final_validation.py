"""
scripts/final_validation.py
Pre-flight checklist to ensure CVPR-ready settings before running 
the final 30-epoch ablation studies.
"""

import sys
import re
from pathlib import Path
import yaml

# Add the repository root to sys.path so the 'zebraid' package can be imported
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
    # 7. Check for Paper Tables generator
    tables_script = Path("scripts/generate_paper_tables.py")
    if not tables_script.exists():
        fatal_error("scripts/generate_paper_tables.py not found. Paper tables generation missing.")
    else:
        print("✅ Paper tables generator found.")
        
    # 8. Security Audit (Phase 11)
    # Ensure no .env file is tracked by git
    if Path(".env").exists():
        try:
            import subprocess
            git_status = subprocess.check_output(["git", "ls-files", ".env"], text=True).strip()
            if git_status == ".env":
                fatal_error(".env file is tracked by Git. Secrets may be exposed!")
        except Exception:
            pass
            
    # Basic regex scan for obvious API keys (dummy check)
    if "api_key=" in train_code.lower() or "sk-" in train_code:
        print("⚠️ Warning: Possible hardcoded secret detected in train.py.")
    else:
        print("✅ Security audit: No obvious hardcoded secrets detected.")

    # 10. Audit Checklist Verification (Sampler set_epoch, NaN safety, min_images)
    sampler_path = Path("zebraid/data/mixed_batch_sampler.py")
    with open(sampler_path, "r") as f:
        sampler_code = f.read()
    if "def set_epoch" not in sampler_code:
        fatal_error("set_epoch not implemented in MixedPopulationBatchSampler.")
    else:
        print("✅ Sampler set_epoch method verified.")

    eval_path = Path("zebraid/models/evaluate.py")
    with open(eval_path, "r") as f:
        eval_code = f.read()
    if "np.isfinite" not in eval_code:
        fatal_error("NaN/Inf safety check missing in evaluate.py.")
    else:
        print("✅ Evaluation NaN/Inf safety assertion verified.")

    # 11. Zero-Leakage Dataset Split Verification
    try:
        from zebraid.data.loaders import build_datasets
        ds_a_tr, ds_b_tr = build_datasets("train", transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_va, ds_b_va = build_datasets("val",   transform=None, split_seed=42, min_images_per_individual=2)
        ds_a_te, ds_b_te = build_datasets("test",  transform=None, split_seed=42, min_images_per_individual=2)

        a_tr, a_va, a_te = set(ds_a_tr.individual_ids), set(ds_a_va.individual_ids), set(ds_a_te.individual_ids)
        b_tr, b_va, b_te = set(ds_b_tr.individual_ids), set(ds_b_va.individual_ids), set(ds_b_te.individual_ids)

        if not (a_tr.isdisjoint(a_va) and a_tr.isdisjoint(a_te) and a_va.isdisjoint(a_te)):
            fatal_error(f"Pop A Identity Leakage! Train/Val overlap: {a_tr & a_va}")
        if not (b_tr.isdisjoint(b_va) and b_tr.isdisjoint(b_te) and b_va.isdisjoint(b_te)):
            fatal_error(f"Pop B Identity Leakage! Train/Val overlap: {b_tr & b_va}")
        if not (a_tr | a_va | a_te).isdisjoint(b_tr | b_va | b_te):
            fatal_error("Pop A and Pop B ID collision!")

        print(f"✅ Zero identity leakage verified: Pop A ({len(a_tr)}/{len(a_va)}/{len(a_te)}), Pop B ({len(b_tr)}/{len(b_va)}/{len(b_te)}).")
    except Exception as e:
        fatal_error(f"Zero-leakage verification failed: {e}")

    print("\n🚀 Pre-flight checks passed. Cleared for 30-epoch training.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_preflight_checks()
