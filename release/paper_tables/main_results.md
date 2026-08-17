# ZebraID — Main Experimental Results

Held-out test split evaluation (N=821 queries across 156 Plains Zebra identities; N=130 queries across 13 Grevy's Zebra identities; split_seed=42; zero data leakage).

| Method | Pop A Rank-1 | Pop A Rank-5 | Pop A Rank-10 | Pop A mAP | Pop B Rank-1 | Pop B Rank-5 | Pop B Rank-10 | Pop B mAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline X** | 89.40% | 94.28% | **96.47%** | **66.84%** | 46.92% | 73.08% | 82.31% | 31.77% |
| **ZebraID (Ours)** | **89.48 ± 1.11%** | 93.99 ± 0.25% | 95.45 ± 0.19% | 65.33 ± 0.53% | **51.54 ± 3.08%** | **75.38 ± 3.53%** | **82.31 ± 2.77%** | **34.50 ± 1.51%** |
| **Delta ZebraID vs Baseline X** | +0.08 pp | -0.28 pp | -1.02 pp | -1.51 pp | **+4.62 pp** | **+2.31 pp** | +0.00 pp | **+2.72 pp** |

*Note: Baseline X and Baseline A represent the exact same underlying model (trained on Population A only) evaluated on held-out test splits. Metric values for ZebraID are reported as sample Mean ± Standard Deviation over three training seeds (42, 43, 44).*
