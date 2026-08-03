#!/usr/bin/env python3
"""
scripts/validate_config.py
Validates `configs/default.yaml` against a strict expected schema.
"""

import sys
from pathlib import Path
import yaml

EXPECTED_SCHEMA = {
    "paths": {"data_root", "processed_root", "checkpoint_dir", "results_dir"},
    "data": {"use_horizontal_flip"},
    "datasets": {"population_a", "population_b", "split_seed", "train_ratio", "val_ratio", "test_ratio"},
    "model": {"primary_backbone", "ablation_backbone", "embedding_dim"},
    "zhash": {"deploy_size_bits", "benchmark_sizes_bits", "backend", "pq_num_subquantizers"},
    "training": {
        "batch_size", "accum_steps", "num_epochs", "learning_rate", "weight_decay",
        "loss", "triplet_margin", "triplet_mining", "mixed_batch_ratio", "num_train_splits",
        "use_wandb", "wandb_project"
    },
    "matching": {"index_type", "top_k", "confidence_threshold", "scale_sim_sizes"},
    "federation": {"org_a_port", "org_b_port", "coordinator_port", "score_buckets", "rate_limit_queries_per_minute", "api_key_env_var"},
    "edge": {"onnx_opset", "quantize_int8", "benchmark_n_trials", "platform_note"}
}

def validate_config(cfg_path: str = "configs/default.yaml"):
    path = Path(cfg_path)
    if not path.exists():
        print(f"❌ FATAL: Config file {path} not found.")
        sys.exit(1)
        
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print(f"=== Configuration Validation ({path}) ===")
    
    errors = 0
    warnings = 0
    
    # 1. Check for missing or extra sections
    for section in EXPECTED_SCHEMA.keys():
        if section not in cfg:
            print(f"❌ Missing required section: [{section}]")
            errors += 1
            
    for section in cfg.keys():
        if section not in EXPECTED_SCHEMA:
            print(f"⚠️ Unused or unknown section found: [{section}]")
            warnings += 1
            
    # 2. Check keys within sections
    for section, expected_keys in EXPECTED_SCHEMA.items():
        if section not in cfg:
            continue
            
        actual_keys = set(cfg[section].keys())
        
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        
        for k in missing:
            print(f"❌ Missing required key: {section}.{k}")
            errors += 1
            
        for k in extra:
            print(f"⚠️ Unused or unknown key found: {section}.{k}")
            warnings += 1
            
    # 3. Value validations
    if cfg.get("data", {}).get("use_horizontal_flip", True) is True:
        print("❌ Conflicting setting: use_horizontal_flip is True, but zebra stripes are asymmetric.")
        errors += 1
        
    bz = cfg.get("training", {}).get("batch_size", 0)
    acc = cfg.get("training", {}).get("accum_steps", 0)
    if bz * acc < 16:
        print(f"⚠️ Warning: Effective batch size ({bz*acc}) is suspiciously small (<16).")
        warnings += 1
        
    if errors > 0:
        print(f"\n❌ Validation Failed: {errors} errors, {warnings} warnings.")
        sys.exit(1)
    else:
        print(f"\n✅ Validation Passed! (0 errors, {warnings} warnings)")

if __name__ == "__main__":
    validate_config()
