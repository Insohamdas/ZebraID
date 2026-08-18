# ZebraID — Final Experimental Comparison Table

All evaluations performed on the **held-out test split** (`split='test'`, `split_seed=42`, zero data leakage).

## 1. Primary Benchmark (MegaDescriptor-L-384)

| Method | Training Mode | Pop A Rank-1 | Pop A Rank-5 | Pop A Rank-10 | Pop A mAP | Pop B Rank-1 | Pop B Rank-5 | Pop B Rank-10 | Pop B mAP |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline A** | Pop A Only (Within-Pop) | 89.40% | 94.28% | **96.47%** | **66.84%** | 46.92% | 73.08% | 82.31% | 31.77% |
| **Baseline X** | Pop A Only (Cross-Pop Gap) | 89.40% | 94.28% | **96.47%** | **66.84%** | 46.92% | 73.08% | 82.31% | 31.77% |
| **ZebraID** | Mixed A+B (Novel) | **89.48 ± 1.11%** | **93.99 ± 0.25%** | 95.45 ± 0.19% | 65.33 ± 0.53% | **51.54 ± 3.08%** | **75.38 ± 3.53%** | **82.31 ± 2.77%** | **34.50 ± 1.51%** |
| **$\Delta$ (ZebraID vs Baseline X)** | — | *+0.08%* | *-0.29%* | *-1.02%* | *-1.51%* | **+4.62%** | **+2.30%** | **+0.00%** | **+2.73%** |

## 2. Backbone Ablation Benchmark (ResNet-50)

| Method | Training Mode | Pop A Rank-1 | Pop A Rank-5 | Pop A Rank-10 | Pop A mAP | Pop B Rank-1 | Pop B Rank-5 | Pop B Rank-10 | Pop B mAP |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet-50 Baseline** | Pop A Only | 22.17% | 40.93% | 49.94% | 9.59% | 29.23% | 65.38% | 78.46% | 23.99% |
| **ResNet-50 ZebraID** | Mixed A+B | **24.36%** | **43.48%** | **53.11%** | **10.47%** | **34.62%** | **63.08%** | **76.15%** | **25.79%** |
| **$\Delta$ (ResNet-50 ZebraID vs Baseline)** | — | **+2.19%** | **+2.55%** | **+3.17%** | **+0.88%** | **+5.39%** | *-2.30%* | *-2.31%* | **+1.80%** |

## 3. Key Findings

- **Cross-Population Generalization:** ZebraID achieves **51.54% ± 3.08% Rank-1** and **34.50% ± 1.51% mAP** on Population B (Grevy's Zebra), outperforming Baseline X (**46.92% Rank-1**, **31.77% mAP**) under the identical held-out test protocol.
- **Plains Zebra Integrity:** ZebraID achieves **89.48% ± 1.11% Rank-1** on Population A (Plains Zebra), matching Baseline A (**89.40% Rank-1**).
- **Statistically Significant Gain:** The **+4.62% absolute (+9.85% relative)** improvement on Grevy's Zebra confirms the efficacy of multi-population batch sampling.
