# ZebraID Production Model (v1.0.0)

Production-ready, L2-normalized 512-dimensional embedding model for zebra individual re-identification across Plains Zebras (*Equus quagga*) and endangered Grevy's Zebras (*Equus grevyi*).

---

## 1. Model Overview

- **Model Name:** ZebraID Production Embedder (MegaDescriptor-L-384)
- **Model Version:** `v1.0.0`
- **Selected Training Seed:** `Seed 44` (Best Epoch `12`)
- **Backbone:** MegaDescriptor-L-384 (Pretrained on diverse wildlife datasets)
- **Projector:** 2-Layer MLP ($2048 \rightarrow 2048 \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow 512$)
- **Embedding Dimension:** `512` (Unit L2-Normalized: $\|\mathbf{e}\|_2 = 1.0$)
- **Checkpoint Location:** `production/model/best_model.pt`
- **File Size:** 818,769,972 bytes
- **SHA-256 Checksum:** `3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80`

---

## 2. Input Specification & Preprocessing

- **Supported Input:** RGB Image (JPEG, PNG, WebP) or Cropped Zebra Flank
- **Input Resolution:** $384 \times 384$ pixels
- **Normalization:** ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`)
- **Horizontal Flipping:** **STRICTLY DISABLED** (Zebra flank stripe patterns are asymmetric)
- **Inference Pipeline:**
  ```python
  from torchvision import transforms
  from PIL import Image

  transform = transforms.Compose([
      transforms.Resize((384, 384)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  ```

---

## 3. Retrieval Usage

For a query embedding $\mathbf{q} \in \mathbb{R}^{512}$ and gallery database $\mathbf{G} \in \mathbb{R}^{N \times 512}$:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{g}_i) = \mathbf{q}^T \mathbf{g}_i$$
Since embeddings are L2-normalized, cosine similarity is equivalent to Euclidean distance ranking ($d^2 = 2 - 2\mathbf{q}^T\mathbf{g}_i$).

---

## 4. Model Provenance & Verification Metrics

### Selection Protocol
The production checkpoint was selected strictly based on **in-training validation performance** without test-set tuning:
1. **Primary Rule:** Highest Population B (Grevy's Zebra) Validation mAP $\rightarrow$ **Seed 44 (48.18% mAP)** vs. Seed 43 (47.97%) and Seed 42 (47.63%).
2. **Secondary Rule:** Population B Validation Rank-1 $\rightarrow$ **60.00% Rank-1**.
3. **Tertiary Rule:** Population A Validation Retention $\rightarrow$ **90.97% Rank-1, 64.76% mAP**.

### In-Training Validation Performance (Authoritative)
| Population | Validation Rank-1 | Validation mAP |
|---|:---:|:---:|
| **Population A (Plains Zebra)** | **90.97%** | **64.76%** |
| **Population B (Grevy's Zebra)** | **60.00%** | **48.18%** |

### Held-Out Test Performance (*Reference Only*)
> **Note:** Held-out test metrics ($N=821$ Pop A queries, $N=130$ Pop B queries, $\text{split\_seed}=42$, zero leakage) are provided for reference only and were not used during selection.

| Population | Seed 44 Test Rank-1 | Seed 44 Test mAP | Multi-Seed Aggregate |
|---|:---:|:---:|:---:|
| **Population A (Plains Zebra)** | 90.50% | 65.87% | 89.48 $\pm$ 1.11% (mAP: 65.33 $\pm$ 0.53%) |
| **Population B (Grevy's Zebra)** | 54.62% | 36.06% | 51.54 $\pm$ 3.08% (mAP: 34.50 $\pm$ 1.51%) |

---

## 5. Known Limitations & Recommendations

1. **Severe Occlusion (>50%):** Occluded zebra flank stripes reduce retrieval confidence; automated detection thresholding at $\ge 0.45$ is recommended.
2. **Extreme View Angles:** Oblique head-on or tail-on angles should be rejected in favor of broadside flank crops ($30^\circ - 150^\circ$).
3. **Flank Disparity:** Left and right zebra flanks have distinct stripe topologies; cross-flank comparisons should be handled with multi-view identity galleries.
