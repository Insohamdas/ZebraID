"""
zebraid/reporting/experiment_tracker.py
Gathers deterministic environment information for reproducibility.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def _get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "Unknown"


def save_experiment_info(out_dir: Path | str, cfg: Dict[str, Any], command_args: list[str], seed: int, mode: str, backbone: str):
    """
    Saves a comprehensive experiment_info.json and a copy of the config.yaml 
    to the output directory for total reproducibility.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get GPU info if available
    gpu_model = "None"
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_model = "Apple MPS"
        
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _get_git_revision_hash(),
        "python_version": sys.version.split(" ")[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_model": gpu_model,
        "os": platform.system() + " " + platform.release(),
        "command_line_args": " ".join(command_args),
        "random_seed": seed,
        "experiment_mode": mode,
        "backbone": backbone,
        "config_used": "config.yaml",
        "repository_version": "ZebraID v1.0"
    }

    # 2. Dump experiment_info.json
    with open(out_dir / "experiment_info.json", "w") as f:
        json.dump(info, f, indent=4)
        
    # 3. Dump config.yaml
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
        
    print(f"✅ Experiment info and config saved to {out_dir}")
