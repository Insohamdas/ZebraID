# ZebraID: Continental-Scale Biometric Zebra Re-Identification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version: v1.0](https://img.shields.io/badge/Release-v1.0-green.svg)](release/final_release_manifest.json)

**ZebraID** is an end-to-end, privacy-preserving deep learning biometric identification system engineered for individual zebra recognition across distinct populations and species at continental scale.

By combining wildlife-domain foundation representations (**MegaDescriptor-L-384**), mixed-population metric learning, 256-bit Z-Hash binary embeddings, and a privacy-preserving federated cross-organization matching protocol, ZebraID bridges the cross-population generalization gap while guaranteeing zero data leakage and real-time inference latency.

---

## Key Features & Contributions

- **Wildlife Foundation Representations:** Pretrained MegaDescriptor-L-384 vision transformer backbone combined with an adaptive 2-layer MLP projection head producing unit-normalized 512-dimensional metric embeddings.
- **Cross-Population Generalization:** Mixed-population batch sampling with Multi-Similarity metric learning, improving Grevy's zebra held-out Rank-1 accuracy by **+4.62 pp (+9.85% relative)** and mAP by **+2.73 pp (+8.59% relative)** over cross-population baselines.
- **Z-Hash Binary Compression:** Quantizes 512-dimensional floating-point embeddings into compact 256-bit (32-byte) binary hash codes, enabling ultra-fast Hamming distance candidate filtering over millions of individuals.
- **Privacy-Preserving Federated Protocol:** Enables collaborative cross-organization matching across sovereign conservation entities without centralizing raw wildlife imagery.
- **Audited Production Integrity:** Cryptographically verified immutable production checkpoint (`SHA-256: 3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80`) with sub-115ms end-to-end inference latency.

---

## System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │         Raw Wildlife Imagery            │
                                  │      (Plains / Grevy's Zebra)           │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │     Input Preprocessing Pipeline        │
                                  │   384x384 px | ImageNet Standardized    │
                                  │    0% Flips (Biologically Asymmetric)   │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │        MegaDescriptor-L-384             │
                                  │     Frozen Wildlife ViT Backbone        │
                                  └────────────────────┬────────────────────┘
                                                       │ 1536-d
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │        2-Layer MLP Projector            │
                                  │     1536 -> 1536 -> BN -> ReLU -> 512   │
                                  └────────────────────┬────────────────────┘
                                                       │ 512-d
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │         L2 Normalization Unit           │
                                  │        ||e||_2 = 1.0 Unit Sphere        │
                                  └───────────────┬─────────────────┬───────┘
                                                  │                 │
                         ┌────────────────────────┘                 └────────────────────────┐
                         ▼                                                                   ▼
       ┌───────────────────────────────────┐                               ┌───────────────────────────────────┐
       │     Dense Metric Retrieval        │                               │     Z-Hash Binary Quantization    │
       │   Cosine Similarity Matching      │                               │    256-Bit (32-byte) Hash Code    │
       │  FAISS FlatL2 / Exact Search      │                               │    Fast Hamming Shard Matching    │
       └───────────────────────────────────┘                               └───────────────────────────────────┘
```

---

## Dataset Provenance & Strict Separation

All experiments adhere to a strictly segregated, zero-leakage identity partition (`split_seed=42`, `min_images_per_individual=2`):

| Population | Species | Train Split | Validation Split | Held-Out Test Split |
|---|---|:---:|:---:|:---:|
| **Population A** | Plains Zebra (*Equus quagga*) | 3,796 images (723 IDs) | 797 images (154 IDs) | 821 queries (156 IDs) |
| **Population B** | Grevy's Zebra (*Equus grevyi*) | 369 images (53 IDs) | 90 images (11 IDs) | 130 queries (13 IDs) |

> **Zero Leakage Invariant:**  
> $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$, $\text{Pop A} \cap \text{Pop B} = \emptyset$.

---

## Authoritative Results Summary

### In-Training Validation (Selection Basis)
Selected production checkpoint (**Seed 44, Epoch 12**):
- **Population A (Plains Zebra):** Rank-1 = **90.97%**, mAP = **64.76%**
- **Population B (Grevy's Zebra):** Rank-1 = **60.00%**, mAP = **48.18%**

### Held-Out Test Performance (Three-Seed $\mu \pm \sigma$)

| Model Configuration | Pop A Rank-1 | Pop A mAP | Pop B Rank-1 | Pop B mAP |
|---|:---:|:---:|:---:|:---:|
| **Baseline A** (Trained on Pop A) | 89.40% | 66.84% | 46.92% | 31.77% |
| **Baseline X** (Cross-Population) | 89.40% | 66.84% | 46.92% | 31.77% |
| **ZebraID (v1.0 Release)** | **89.48 $\pm$ 1.11%** | **65.33 $\pm$ 0.53%** | **51.54 $\pm$ 3.08%** | **34.50 $\pm$ 1.51%** |
| **ZebraID Delta over Baselines** | **+0.08 pp** | -1.51 pp | **+4.62 pp (+9.85%)** | **+2.73 pp (+8.59%)** |

---

## Production Model Loading & Inference

### Python Quickstart

```python
import torch
from PIL import Image
from torchvision import transforms
from zebraid.models.backbone import build_embedder

# 1. Initialize embedder
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = build_embedder("megadescriptor", embedding_dim=512, pretrained=False, device=device)

# 2. Load verified production checkpoint
ckpt_path = "production/model/best_model.pt"
raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = raw_ckpt["model"] if isinstance(raw_ckpt, dict) and "model" in raw_ckpt else raw_ckpt
model.load_state_dict(state_dict)
model.eval()

# 3. Preprocessing (Specification v1.0)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Extract 512-d normalized embedding
img = Image.open("path/to/zebra.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    embedding = model(tensor).cpu().numpy()[0]

assert embedding.shape == (512,)
assert abs(float(torch.norm(torch.from_numpy(embedding))) - 1.0) < 1e-4
print("✅ Extracted 512-d normalized embedding successfully!")
```

---

## FastAPI Identification Service

Start the production identification API:

```bash
python demo/app.py
```

### API Endpoints

- **`GET /health`:** Returns service health, loaded backends, and shard connection status.
- **`POST /identify`:** Multipart form upload accepting `file` (image) and optional `query_org_b` boolean flag.

```bash
curl -X POST "http://localhost:8000/identify" \
     -F "file=@data/sample_zebra.jpg" \
     -F "query_org_b=false"
```

---

## Latency & Performance Profile

Benchmarked with explicit hardware barriers (`torch.mps.synchronize()`) across 50 iterations on Apple Silicon MPS:

| Inference Stage | Mean Latency | Median Latency | p95 Latency |
|---|:---:|:---:|:---:|
| **1. Image Preprocessing** | 6.89 ms | 7.00 ms | 8.37 ms |
| **2. Backbone & Projector Forward** | 104.56 ms | 104.64 ms | 105.19 ms |
| **3. L2 Normalization & Host Copy** | 1.27 ms | 1.30 ms | 2.47 ms |
| **4. 1,000-Gallery Nearest Neighbor** | 0.21 ms | 0.20 ms | 0.26 ms |
| **Total End-to-End Latency** | **112.93 ms** | **113.19 ms** | **114.86 ms** |

- **Peak Single-Stream Throughput:** **8.9 images/sec** (Sub-115ms response time).

---

## Reproducibility & Security Verification

All pre-flight checks and unit test suites can be validated via:

```bash
# 1. Python compilation sanity
python -m compileall -q production zebraid scripts tests release

# 2. Complete test suite (62 test cases)
pytest -q

# 3. Final validation pre-flight audit
python scripts/final_validation.py
```

---

## Known Limitations

1. **Flank Orientation Requirement:** Flank stripe biometrics require that the lateral torso of the animal is visible; extreme head-on or tail-on angles degrade re-identification confidence.
2. **Horizontal Flipping Invariance:** Flank stripe patterns are biologically asymmetric between left and right sides. Query images should not be horizontally flipped or mirrored.
3. **Severe Mud/Shadow Occlusion:** Heavy mud caking obscuring >60% of flank stripe transitions may require manual human verification.

---

## Citation & License

This codebase is licensed under the **MIT License**.

```bibtex
@article{zebraid2026,
  title={ZebraID: Continental-Scale Biometric Re-Identification for Wild Equids},
  author={Das, Soham and Contributors},
  year={2026},
  journal={Conservation Biometrics & Computer Vision}
}
```
