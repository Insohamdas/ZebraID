# ZebraID — Paper Table & Statistical Summary Generation Report

**Execution Scope:** Final Publication Table Generation & Statistical Audit  
**Data Provenance:** Strict Held-Out Test ($N=821$ Pop A queries, $N=130$ Pop B queries, split_seed=42) and Validation Splits  
**Zero Leakage Status:** Fully Audited & Verified ✅  

---

## 1. Source Files Used

The following authoritative repository artifacts were cross-checked and used as exact data sources:
- `results/test_metrics.json`: Aggregated and multi-seed held-out test evaluation metrics.
- `results/baseline_test_results.json`: Evaluated Baseline A & X MegaDescriptor and ResNet-50 metrics.
- `results/research_release_manifest.json`: Frozen research release metadata and hashes.
- `release/validation/validation_summary.json`: Multi-seed validation metrics.
- `checkpoints/zebraid/megadescriptor/seed*/test_metrics.json`: Per-seed test evaluation outputs.

---

## 2. Values Extracted & Verified

- **Validation Split (Pop A / Pop B):**
  - Seed 42: Pop A R-1 = 91.22%, mAP = 64.40% | Pop B R-1 = 61.11%, mAP = 47.63%
  - Seed 43: Pop A R-1 = 91.34%, mAP = 64.00% | Pop B R-1 = 62.22%, mAP = 47.97%
  - Seed 44: Pop A R-1 = 90.97%, mAP = 64.76% | Pop B R-1 = 60.00%, mAP = 48.18%
  - **Validation Aggregate:** Pop A R-1 = 91.18 ± 0.19%, mAP = 64.39 ± 0.38% | Pop B R-1 = 61.11 ± 1.11%, mAP = 47.93 ± 0.28%
- **Held-Out Test Split (Pop A / Pop B):**
  - Baseline X (Pop-A model): Pop A R-1 = 89.40%, mAP = 66.84% | Pop B R-1 = 46.92%, mAP = 31.77%
  - ZebraID Seed 42: Pop A R-1 = 88.31%, mAP = 64.80% | Pop B R-1 = 48.46%, mAP = 34.38%
  - ZebraID Seed 43: Pop A R-1 = 89.65%, mAP = 65.32% | Pop B R-1 = 51.54%, mAP = 33.06%
  - ZebraID Seed 44: Pop A R-1 = 90.50%, mAP = 65.87% | Pop B R-1 = 54.62%, mAP = 36.06%
  - **ZebraID Test Aggregate:** Pop A R-1 = **89.48 ± 1.11%**, mAP = **65.33 ± 0.53%** | Pop B R-1 = **51.54 ± 3.08%**, mAP = **34.50 ± 1.51%**

---

## 3. Comparison Metrics Calculated

- **Population B (Grevy's Zebra):**
  - Absolute Rank-1 Difference: **+4.62 percentage points** (51.54% vs. 46.92%)
  - Relative Rank-1 Improvement: **+9.85%**
  - Absolute mAP Difference: **+2.73 percentage points** (34.50% vs. 31.77%)
  - Relative mAP Improvement: **+8.59%**
- **Population A (Plains Zebra):**
  - Absolute Rank-1 Difference: **+0.08 percentage points** (89.48% vs. 89.40%)
  - Absolute mAP Difference: **-1.51 percentage points** (65.33% vs. 66.84%)

---

## 4. Generated Artifacts Inventory

| Output Path | Description | Format |
|---|---|:---:|
| `results/final_validation_results.csv` | Per-seed validation results | CSV |
| `results/final_test_results.csv` | Per-seed and baseline test results | CSV |
| `results/final_three_seed_summary.csv` | Complete multi-seed statistical summary | CSV |
| `paper_tables/main_results.md` | Main paper comparison table | Markdown |
| `paper_tables/main_results.csv` | Main paper comparison table | CSV |
| `paper_tables/main_results.tex` | Publication-ready CVPR/IEEE table | LaTeX |
| `paper_tables/zebraid_three_seed_validation.tex` | Three-seed validation table | LaTeX |
| `paper_tables/zebraid_three_seed_test.tex` | Three-seed held-out test table | LaTeX |
| `paper_tables/backbone_ablation.md` | ResNet-50 vs MegaDescriptor ablation | Markdown |
| `paper_tables/backbone_ablation.tex` | ResNet-50 vs MegaDescriptor ablation | LaTeX |
| `paper_tables/figures/pop_b_rank1_comparison.png` | Pop-B Rank-1 comparison chart | PNG (300 DPI) |
| `paper_tables/figures/pop_b_map_comparison.png` | Pop-B mAP comparison chart | PNG (300 DPI) |
| `paper_tables/figures/three_seed_variability.png` | Multi-seed stability plot | PNG (300 DPI) |
| `paper_tables/figures/pop_a_vs_pop_b_comparison.png` | Cross-population summary bar chart | PNG (300 DPI) |
| `results/final_results_summary.md` | Authoritative summary document | Markdown |
| `results/paper_table_generation_report.md` | Generation audit report | Markdown |

---

## 5. Quality & Consistency Audit Status

- **Discrepancies Encountered:** Zero (0).
- **LaTeX Compilation Check:** Passed (valid syntax, proper `\pm`, escaped underscores, no missing values).
- **Release Directory Parity:** 100% agreement with `results/research_release_manifest.json`.
