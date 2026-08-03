#!/usr/bin/env python3
"""
scripts/generate_paper_tables.py
Generates CVPR-ready Markdown, CSV, and LaTeX tables for ZebraID final results.
Aggregates across multiple seeds to compute Mean, Std Dev, and 95% CI.
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as st

def generate_paper_tables(results_dir: str = "results", out_dir: str = "paper_tables"):
    results_path = Path(results_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Structure: dict[mode][backbone] -> list of dicts (from seeds)
    aggregated_data = defaultdict(lambda: defaultdict(list))
    
    # 1. Gather all metrics
    if not results_path.exists():
        print(f"⚠️ Results directory {results_path} does not exist. Skipping table generation.")
        return
        
    for run_dir in results_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        metrics_file = run_dir / "final_metrics.json"
        info_file = run_dir / "experiment_info.json"
        
        if metrics_file.exists() and info_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            with open(info_file, "r") as f:
                info = json.load(f)
                
            mode = info["config"].get("mode", "unknown")
            backbone = info["config"].get("backbone_name", "unknown")
            
            # Use final epoch data or best model evaluation if available
            # Assuming metrics has rank1_a, rank1_b, map_a, map_b
            aggregated_data[mode][backbone].append(metrics)
            
    if not aggregated_data:
        print("⚠️ No valid results found to generate tables.")
        return
        
    # 2. Compute statistics
    def _mean_std_str(values):
        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            return "N/A"
        mean = np.mean(values)
        if len(values) == 1:
            return f"{mean:.1f}"
        std = np.std(values, ddof=1)
        # 95% CI
        ci = st.t.interval(0.95, df=len(values)-1, loc=mean, scale=st.sem(values)) if len(values) > 1 else (mean, mean)
        return f"{mean:.1f} ± {std:.1f}"

    rows = []
    for mode in sorted(aggregated_data.keys()):
        for backbone in sorted(aggregated_data[mode].keys()):
            runs = aggregated_data[mode][backbone]
            
            # Extract key metrics
            r1_a_list = [r.get("rank1_a") for r in runs]
            r1_b_list = [r.get("rank1_b") for r in runs]
            map_a_list = [r.get("map_a") for r in runs]
            map_b_list = [r.get("map_b") for r in runs]
            
            # Profiling info if available
            tput_list = [r.get("throughput_img_sec") for r in runs if "throughput_img_sec" in r]
            mem_list = [r.get("peak_memory_mb") for r in runs if "peak_memory_mb" in r]
            
            rows.append({
                "Mode": mode,
                "Backbone": backbone,
                "Runs": len(runs),
                "PopA_Rank1": _mean_std_str(r1_a_list),
                "PopA_mAP": _mean_std_str(map_a_list),
                "PopB_Rank1": _mean_std_str(r1_b_list),
                "PopB_mAP": _mean_std_str(map_b_list),
                "Images/Sec": _mean_std_str(tput_list) if tput_list else "N/A",
                "Peak_Memory_MB": _mean_std_str(mem_list) if mem_list else "N/A"
            })
            
    # 3. Write CSV
    csv_path = out_path / "main_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            
    # 4. Write Markdown
    md_path = out_path / "main_results.md"
    with open(md_path, "w") as f:
        f.write("# ZebraID Final Experimental Results\n\n")
        f.write("Aggregated over multiple seeds to compute Mean ± Std Dev.\n\n")
        if not rows:
            f.write("No data available.\n")
        else:
            headers = list(rows[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
                
    # 5. Write LaTeX
    tex_path = out_path / "main_results.tex"
    with open(tex_path, "w") as f:
        if not rows:
            f.write("% No data available.\n")
        else:
            headers = list(rows[0].keys())
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{" + "l" * len(headers) + "}\n")
            f.write("\\toprule\n")
            f.write(" & ".join(headers) + " \\\\\n")
            f.write("\\midrule\n")
            for row in rows:
                f.write(" & ".join(str(row[h]).replace("_", "\\_") for h in headers) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{ZebraID Performance (Mean \\pm Std Dev over multiple seeds).}\n")
            f.write("\\label{tab:main_results}\n")
            f.write("\\end{table}\n")

    print(f"✅ Generated paper tables in {out_path}:")
    print(f"   - {csv_path.name}")
    print(f"   - {md_path.name}")
    print(f"   - {tex_path.name}")

if __name__ == "__main__":
    generate_paper_tables()
