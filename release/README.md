# ZebraID v1.0 — Research & Production Release Package

**Release Version:** `v1.0`  
**Release Git Commit:** `b4173c962345b7057b78b824c83aeb93e0de20eb`  
**Selected Production Model:** `Seed 44 (Epoch 12)`  
**Checkpoint SHA-256:** `3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80`  

---

## Directory Organization

```
release/
├── final_release_manifest.json     # Complete authoritative cryptographic release manifest
├── final_release_report.md         # Comprehensive release audit and benchmark report
├── README.md                       # Release overview and getting started guide
├── code_metadata/                  # Frozen configurations and release manifests
├── checkpoints/                    # Cryptographic catalog of all multi-seed model weights
├── validation/                     # Multi-seed validation metrics and logs
├── test/                           # Held-out test evaluation outputs and comparisons
├── production/                     # Production model metadata, specifications, and README
├── paper_tables/                   # Authoritative publication tables (LaTeX, CSV, Markdown)
└── reports/                        # Full evaluation and latency methodology reports
```

---

## Quick Start — Production Inference

```python
import torch
from PIL import Image
from torchvision import transforms
from zebraid.models.backbone import build_embedder

# 1. Initialize backbone & projector
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = build_embedder("megadescriptor", embedding_dim=512, pretrained=False, device=device)

# 2. Load frozen production weights
ckpt = torch.load("production/model/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

# 3. Preprocess & extract L2-normalized 512-d embedding
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("zebra.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    embedding = model(tensor).cpu().numpy()[0]  # (512,) ||e||_2 = 1.0
```
