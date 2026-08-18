# ZebraID — Final Held-Out Test Evaluation Report

**Evaluation Mode:** Held-Out Unseen Test Split (`split='test'`)
**Backbone:** `megadescriptor` (MegaDescriptor-L-384)
**Evaluated Seeds:** `[42, 43, 44]`
**Fixed Split Seed:** `42` (Zero Data Leakage Guaranteed ✅)

## 1. Zero Identity Leakage Audit

| Verification Check | Status | Overlap Count |
|---|---|---|
| **Pop A: Train ∩ Test** | PASS ✅ | 0 |
| **Pop A: Val ∩ Test** | PASS ✅ | 0 |
| **Pop B: Train ∩ Test** | PASS ✅ | 0 |
| **Pop B: Val ∩ Test** | PASS ✅ | 0 |
| **Cross-Population: Pop A ∩ Pop B** | PASS ✅ | 0 |

## 2. Test Split Demographics

| Attribute | Population A (Plains Zebra) | Population B (Grevy's Zebra) |
|---|---|---|
| **Held-Out Test Identities** | 156 | 13 |
| **Total Test Queries ($N$)** | 821 | 130 |
| **Valid Queries (Multi-Image)** | 821 | 130 |
| **Singleton Queries** | 0 | 0 |

## 3. Final Multi-Seed Test Results (Mean ± Std Dev, 95% CI)

### Population A — Plains Zebra (GZGC)
| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std Dev | 95% CI |
|---|---|---|---|---|---|
| **Rank-1** | 88.31% | 89.65% | 90.50% | **89.48% ± 1.11%** | [86.74%, 92.23%] |
| **Rank-5** | 93.79% | 94.28% | 93.91% | **93.99% ± 0.25%** | [93.36%, 94.62%] |
| **Rank-10** | 95.25% | 95.62% | 95.49% | **95.45% ± 0.19%** | [94.99%, 95.91%] |
| **mAP** | 64.80% | 65.32% | 65.87% | **65.33% ± 0.53%** | [64.01%, 66.65%] |

### Population B — Grevy's Zebra (Mpala)
| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std Dev | 95% CI |
|---|---|---|---|---|---|
| **Rank-1** | 48.46% | 51.54% | 54.62% | **51.54% ± 3.08%** | [43.89%, 59.18%] |
| **Rank-5** | 76.15% | 71.54% | 78.46% | **75.38% ± 3.53%** | [66.63%, 84.14%] |
| **Rank-10** | 83.08% | 79.23% | 84.62% | **82.31% ± 2.77%** | [75.42%, 89.20%] |
| **mAP** | 34.38% | 33.06% | 36.06% | **34.50% ± 1.51%** | [30.76%, 38.24%] |

## 4. Multi-Image Queries Sub-Analysis

| Population | Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std Dev |
|---|---|---|---|---|---|---|
| **Pop A (Plains)** | Rank-1 | 88.31% | 89.65% | 90.50% | 89.48% ± 1.11% |
| **Pop A (Plains)** | mAP | 64.80% | 65.32% | 65.87% | 65.33% ± 0.53% |
| **Pop B (Grevy's)** | Rank-1 | 48.46% | 51.54% | 54.62% | 51.54% ± 3.08% |
| **Pop B (Grevy's)** | mAP | 34.38% | 33.06% | 36.06% | 34.50% ± 1.51% |

## 5. Visual Artifacts & Diagnostic Plots

![Aggregated Test Performance](aggregated_test_performance.png)

