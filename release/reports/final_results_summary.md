# ZebraID v1.0 — Authoritative Research Results & Statistical Summary

---

## 1. Dataset & Split Provenance

- **Fixed Split Seed:** `42`
- **Identity Filtering:** Minimum 2 images per individual (`min_images_per_individual=2`), eliminating singleton queries.
- **Data Splits:** Uniform 70% Train / 15% Validation / 15% Test partitioning.
- **Population A (GZGC Plains Zebra, *Equus quagga*):**
  - Train: 3,796 images across 723 identities
  - Validation: 797 images across 154 identities
  - Test: 821 queries across 156 identities
  - Total Eligible: 5,414 images across 1,033 identities
- **Population B (Labeled Mpala Grevy's Zebra, *Equus grevyi*):**
  - Train: 369 images across 53 identities
  - Validation: 90 images across 11 identities
  - Test: 130 queries across 13 identities
  - Total Eligible: 589 images across 77 identities
- **Leakage Audit:**
  - Pop A: Train, Val, and Test sets are mutually disjoint (0 identity overlap).
  - Pop B: Train, Val, and Test sets are mutually disjoint (0 identity overlap).
  - Cross-Population: Zero identity overlap between Pop A and Pop B.

---

## 2. In-Training Validation Performance (Three Seeds)

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Sample Std Dev | Min / Max | Seed-Level 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pop A Rank-1** | 91.22% | 91.34% | 90.97% | **91.18% ± 0.19%** | [90.97%, 91.34%] | [90.70%, 91.65%] |
| **Pop A mAP** | 64.40% | 64.00% | 64.76% | **64.38% ± 0.38%** | [64.00%, 64.76%] | [63.44%, 65.33%] |
| **Pop B Rank-1** | 61.11% | 62.22% | 60.00% | **61.11% ± 1.11%** | [60.00%, 62.22%] | [58.35%, 63.87%] |
| **Pop B mAP** | 47.63% | 47.97% | 48.18% | **47.93% ± 0.28%** | [47.63%, 48.18%] | [47.24%, 48.61%] |

---

## 3. Held-Out Test Performance (Three Seeds)

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Sample Std Dev | Min / Max | Seed-Level 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pop A Rank-1** | 88.31% | 89.65% | 90.50% | **89.48% ± 1.11%** | [88.31%, 90.50%] | [86.74%, 92.23%] |
| **Pop A Rank-5** | 93.79% | 94.28% | 93.91% | **93.99% ± 0.25%** | [93.79%, 94.28%] | [93.36%, 94.62%] |
| **Pop A Rank-10** | 95.25% | 95.62% | 95.49% | **95.45% ± 0.19%** | [95.25%, 95.62%] | [94.99%, 95.91%] |
| **Pop A mAP** | 64.80% | 65.32% | 65.87% | **65.33% ± 0.53%** | [64.80%, 65.87%] | [64.01%, 66.65%] |
| **Pop B Rank-1** | 48.46% | 51.54% | 54.62% | **51.54% ± 3.08%** | [48.46%, 54.62%] | [43.89%, 59.18%] |
| **Pop B Rank-5** | 76.15% | 71.54% | 78.46% | **75.38% ± 3.53%** | [71.54%, 78.46%] | [66.63%, 84.14%] |
| **Pop B Rank-10** | 83.08% | 79.23% | 84.62% | **82.31% ± 2.77%** | [79.23%, 84.62%] | [75.42%, 89.20%] |
| **Pop B mAP** | 34.38% | 33.06% | 36.06% | **34.50% ± 1.51%** | [33.06%, 36.06%] | [30.76%, 38.24%] |

---

## 4. Comparison Against Baseline X

Baseline X evaluates the in-domain Pop-A-trained MegaDescriptor-L-384 model on held-out test splits without cross-population exposure.

| Population | Metric | Baseline X | ZebraID (Three-Seed Mean ± Std) | Absolute Difference | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Population B (Grevy's)** | **Rank-1** | 46.92% | **51.54 ± 3.08%** | **+4.62 percentage points** | **+9.84%** |
| **Population B (Grevy's)** | **mAP** | 31.77% | **34.50 ± 1.51%** | **+2.72 percentage points** | **+8.58%** |
| **Population A (Plains)** | **Rank-1** | 89.40% | **89.48 ± 1.11%** | **+0.08 percentage points** | — |
| **Population A (Plains)** | **mAP** | 66.84% | **65.33 ± 0.53%** | **-1.51 percentage points** | — |

---

## 5. Main Research Finding

**Conservative Synthesis:**
> Under identical held-out test protocol (split_seed=42, zero identity leakage), **ZebraID improves held-out Population B (Grevy's Zebra) identification over the Pop-A-trained baseline** (+4.62 percentage points Rank-1, +2.73 percentage points mAP) while **largely preserving Population A (Plains Zebra) Rank-1 performance** (+0.08 percentage points, within one standard deviation).

*Note on Statistical Terminology: Interval metrics reflect seed-level estimation across 3 independent training seeds (N=3). Formal hypothesis claims are constrained strictly to observed metric deltas without overgeneralized claims of universal optimality.*
