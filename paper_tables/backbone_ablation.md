# ZebraID — Backbone Architecture Ablation

Comparing the ImageNet-pretrained standard ResNet-50 against the wildlife-domain MegaDescriptor-L-384 on held-out test splits (split_seed=42).

| Model | Training Mode | Pop A Rank-1 | Pop A mAP | Pop B Rank-1 | Pop B mAP |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **ResNet-50 Baseline** | Pop A Only | 22.17% | 9.59% | 29.23% | 23.99% |
| **ResNet-50 ZebraID** | Mixed Pop A + B | 24.36% | 10.47% | 34.62% | 25.79% |
| **MegaDescriptor ZebraID (Ours)** | Mixed Pop A + B | **89.48 ± 1.11%** | **65.33 ± 0.53%** | **51.54 ± 3.08%** | **34.50 ± 1.51%** |

*Key Takeaway: MegaDescriptor-L-384 provides a +65.12 pp Rank-1 improvement on Plains Zebra and +16.92 pp on Grevy's Zebra over ResNet-50 ZebraID, demonstrating the critical importance of wildlife-specific pretraining.*
