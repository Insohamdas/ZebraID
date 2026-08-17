#!/usr/bin/env python3
"""
scripts/build_research_release.py
Builds the immutable research release manifest and release directory structure:
  - results/research_release_manifest.json
  - release/ (code_metadata, checkpoints, validation, test, paper_tables, reports)
"""

import json
import shutil
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RELEASE_DIR = REPO_ROOT / "release"

# ── 1. Construct Immutable Manifest ───────────────────────────────────────────
manifest = {
    "release_name": "ZebraID Research Release v1.0",
    "release_date": "2026-08-17T18:00:00Z",
    "git_metadata": {
        "release_commit": "80566ad21baa6e7c550884598357b5a9ebc7c49f",
        "branch": "main",
        "seed_commits": {
            "seed42": {
                "commit": "7f3ae22d17f8e810c0f6015afac13010dc54ed83",
                "description": "fix(pipeline): implement audit fixes for sampler, singletons, NaN safety, and dual evaluation metrics"
            },
            "seed43": {
                "commit": "8ba24ef0d3aca48cc394c153fbf05c816332ad65",
                "description": "fix(data): decouple training seed from dataset split seed for multi-seed invariance"
            },
            "seed44": {
                "commit": "8ba24ef0d3aca48cc394c153fbf05c816332ad65",
                "description": "fix(data): decouple training seed from dataset split seed for multi-seed invariance"
            }
        }
    },
    "environment": {
        "python_version": "3.11.7 (MPS) / 3.12.13 (CUDA)",
        "pytorch_version": "2.13.0 (MPS) / 2.10.0+cu128 (CUDA)",
        "cuda_version": "12.8 (Tesla T4) / N/A (Apple MPS)",
        "gpu_models": ["Apple MPS (Mac mini M2 Pro)", "NVIDIA Tesla T4 (Google Cloud)"]
    },
    "dataset_provenance": {
        "split_seed": 42,
        "min_images_per_individual": 2,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "population_a_gzgc_plains": {
            "species": "Equus quagga (Plains Zebra)",
            "train": {"images": 3796, "identities": 723},
            "val": {"images": 797, "identities": 154},
            "test": {"images": 821, "identities": 156},
            "total_eligible": {"images": 5414, "identities": 1033}
        },
        "population_b_mpala_grevys": {
            "species": "Equus grevyi (Grevy's Zebra)",
            "train": {"images": 369, "identities": 53},
            "val": {"images": 90, "identities": 11},
            "test": {"images": 130, "identities": 13},
            "total_eligible": {"images": 589, "identities": 77}
        },
        "leakage_audit": {
            "pop_a_train_val_overlap": 0,
            "pop_a_train_test_overlap": 0,
            "pop_a_val_test_overlap": 0,
            "pop_b_train_val_overlap": 0,
            "pop_b_train_test_overlap": 0,
            "pop_b_val_test_overlap": 0,
            "cross_population_overlap": 0,
            "zero_leakage_verified": True
        }
    },
    "checkpoints": {
        "zebraid_megadescriptor_seed42": {
            "path": "checkpoints/zebraid/megadescriptor/seed42/best_model.pt",
            "size_bytes": 818765172,
            "sha256": "2ed0761ba366fe33d290ea06afbd36ef0780635753eb7475abd75b2f49f8b68a",
            "best_epoch": 3,
            "training_seed": 42,
            "split_seed": 42,
            "commit": "7f3ae22d17f8e810c0f6015afac13010dc54ed83",
            "val_metrics": {"rank1_a": 0.92095, "map_a": 0.64546, "rank1_b": 0.58889, "map_b": 0.45815},
            "test_metrics": {"rank1_a": 0.88307, "map_a": 0.64801, "rank1_b": 0.48462, "map_b": 0.34376}
        },
        "zebraid_megadescriptor_seed43": {
            "path": "checkpoints/zebraid/megadescriptor/seed43/best_model.pt",
            "size_bytes": 818769972,
            "sha256": "a0df3a25f62aa1524131fbb3d6b8b37815ebc0ab95a3b1ec8f36f4e9bc62ffb7",
            "best_epoch": 8,
            "training_seed": 43,
            "split_seed": 42,
            "commit": "8ba24ef0d3aca48cc394c153fbf05c816332ad65",
            "val_metrics": {"rank1_a": 0.91343, "map_a": 0.64001, "rank1_b": 0.62222, "map_b": 0.47974},
            "test_metrics": {"rank1_a": 0.89647, "map_a": 0.65324, "rank1_b": 0.51538, "map_b": 0.33056}
        },
        "zebraid_megadescriptor_seed44": {
            "path": "checkpoints/zebraid/megadescriptor/seed44/best_model.pt",
            "size_bytes": 818769972,
            "sha256": "3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80",
            "best_epoch": 12,
            "training_seed": 44,
            "split_seed": 42,
            "commit": "8ba24ef0d3aca48cc394c153fbf05c816332ad65",
            "val_metrics": {"rank1_a": 0.90966, "map_a": 0.64758, "rank1_b": 0.60000, "map_b": 0.48180},
            "test_metrics": {"rank1_a": 0.90499, "map_a": 0.65866, "rank1_b": 0.54615, "map_b": 0.36059}
        },
        "baseline_a_megadescriptor_seed42": {
            "path": "checkpoints/baseline_a/megadescriptor/seed42/best_model.pt",
            "size_bytes": 793544353,
            "sha256": "ee07dcf7ef3da6fd31f711d582820f4cc810b342c0f24d696ba233ee0eefd12e",
            "best_epoch": 16,
            "training_seed": 42,
            "split_seed": 42,
            "commit": "30a79eac054ae287b8bce39f92f6ae141aec3af4",
            "test_metrics": {"rank1_a": 0.89403, "map_a": 0.66844, "rank1_b": 0.46923, "map_b": 0.31772}
        },
        "zebraid_resnet50_seed42": {
            "path": "checkpoints/zebraid/resnet50/seed42/best_model.pt",
            "size_bytes": 115370776,
            "sha256": "70364ba6213cb295aeea5fbf0de36f189f60e5c4db2018055450bec0e4e25435",
            "training_seed": 42,
            "split_seed": 42,
            "test_metrics": {"rank1_a": 0.24361, "map_a": 0.10470, "rank1_b": 0.34615, "map_b": 0.25794}
        },
        "baseline_a_resnet50_seed42": {
            "path": "checkpoints/baseline_a/resnet50/seed42/best_model.pt",
            "size_bytes": 115370776,
            "sha256": "397b0f40eba04d1d2b83587f156372c12de38426fdd5629ccd359c4142bddcf6",
            "training_seed": 42,
            "split_seed": 42,
            "test_metrics": {"rank1_a": 0.22168, "map_a": 0.09595, "rank1_b": 0.29231, "map_b": 0.23988}
        }
    },
    "final_statistical_test_results": {
        "population_a_plains_zebra": {
            "rank1": "89.48 ± 1.11%",
            "rank5": "93.99 ± 0.25%",
            "rank10": "95.45 ± 0.19%",
            "map": "65.33 ± 0.53%"
        },
        "population_b_grevys_zebra": {
            "rank1": "51.54 ± 3.08%",
            "rank5": "75.38 ± 3.53%",
            "rank10": "82.31 ± 2.77%",
            "map": "34.50 ± 1.51%"
        },
        "baseline_comparison": {
            "pop_b_rank1_gain_absolute": "+4.62%",
            "pop_b_rank1_gain_relative": "+9.85%",
            "pop_b_map_gain_absolute": "+2.73%",
            "pop_b_map_gain_relative": "+8.59%"
        }
    }
}

manifest_path = RESULTS_DIR / "research_release_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Generated: {manifest_path}")


# ── 2. Construct release/ Directory Structure ──────────────────────────────────
subdirs = [
    RELEASE_DIR / "code_metadata",
    RELEASE_DIR / "checkpoints",
    RELEASE_DIR / "validation",
    RELEASE_DIR / "test",
    RELEASE_DIR / "paper_tables",
    RELEASE_DIR / "reports",
]

for sd in subdirs:
    sd.mkdir(parents=True, exist_ok=True)

# Copy metadata & configs
shutil.copy(manifest_path, RELEASE_DIR / "code_metadata" / "research_release_manifest.json")
if (REPO_ROOT / "configs" / "default.yaml").exists():
    shutil.copy(REPO_ROOT / "configs" / "default.yaml", RELEASE_DIR / "code_metadata" / "config_frozen.yaml")

# Checkpoint catalog (hashes and paths, preventing file bloat)
checkpoint_catalog = {
    "manifest_version": "1.0",
    "description": "Immutable catalog of trained model weights and SHA-256 hashes",
    "models": manifest["checkpoints"]
}
with open(RELEASE_DIR / "checkpoints" / "checkpoint_catalog.json", "w") as f:
    json.dump(checkpoint_catalog, f, indent=2)

# Validation artifacts
val_summary = {
    "seed42": manifest["checkpoints"]["zebraid_megadescriptor_seed42"]["val_metrics"],
    "seed43": manifest["checkpoints"]["zebraid_megadescriptor_seed43"]["val_metrics"],
    "seed44": manifest["checkpoints"]["zebraid_megadescriptor_seed44"]["val_metrics"],
}
with open(RELEASE_DIR / "validation" / "validation_summary.json", "w") as f:
    json.dump(val_summary, f, indent=2)

# Test artifacts
for f_name in ["test_metrics.json", "baseline_test_results.json"]:
    src = RESULTS_DIR / f_name
    if src.exists():
        shutil.copy(src, RELEASE_DIR / "test" / f_name)

# Paper tables
for f_name in ["final_comparison.csv", "final_comparison.md", "final_comparison.tex"]:
    src = RESULTS_DIR / f_name
    if src.exists():
        shutil.copy(src, RELEASE_DIR / "paper_tables" / f_name)

# Reports
for f_name in ["baseline_test_report.md", "test_report.md"]:
    src = RESULTS_DIR / f_name
    if src.exists():
        shutil.copy(src, RELEASE_DIR / "reports" / f_name)

print("✅ Successfully built and populated release/ directory structure!")
