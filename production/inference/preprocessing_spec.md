# ZebraID Production Preprocessing Specification (v1.0)

**Input Resolution:** $384 \times 384$ pixels (RGB)  
**Aspect Ratio Strategy:** Bilinear resize to fixed square geometry  
**Color Space:** 8-bit RGB normalized to $[0.0, 1.0]$  
**Horizontal Flips:** **STRICTLY DISABLED (0%)** — Biologically asymmetric flank stripe patterns  

---

## Preprocessing Pipeline Stages

1. **Image Ingestion:** Load 8-bit RGB image via PIL or OpenCV. If cropped bounding box is available, crop flank region of interest.
2. **Geometric Normalization:** Resize image to $384 \times 384$ pixels (`interpolation=InterpolationMode.BILINEAR`).
3. **Channel Transposition & Scaling:** Convert to floating-point PyTorch tensor $(C, H, W)$ scaled to range $[0.0, 1.0]$.
4. **Standardization:** Apply ImageNet channel standardization:
   - Mean: `[0.485, 0.456, 0.406]`
   - Standard Deviation: `[0.229, 0.224, 0.225]`
5. **Batch Dimension:** Expand tensor dimension to $(1, 3, 384, 384)$ for model forward pass.

---

## PyTorch Reference Implementation

```python
from torchvision import transforms
from PIL import Image

production_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
