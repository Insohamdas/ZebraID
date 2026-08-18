"""
zebraid/edge/export.py
ONNX export for ZebraEmbedder — hardware-agnostic edge deployment.

Exports the trained model to:
  1. Full-precision ONNX (float32) — for Raspberry Pi / Jetson / Mac mini.
  2. INT8 quantized ONNX — simulates the size/speed optimization that would
     run on an edge inference node.

The same ONNX file runs unchanged on:
  - Mac mini CPU (current test platform — CPU proxy)
  - Raspberry Pi 5 via ONNX Runtime
  - Jetson Orin Nano via ONNX Runtime or TensorRT

Usage:
    python -m zebraid.edge.export \\
        --checkpoint checkpoints/zebraid/megadescriptor/seed42/best_model.pt \\
        --output_dir checkpoints/onnx/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zebraid.models.backbone import build_embedder


def export_onnx(
    checkpoint_path: str,
    output_dir: str,
    backbone_name: str = "megadescriptor",
    embedding_dim: int = 512,
    img_size: int = 384,
    opset: int = 17,
    quantize_int8: bool = True,
) -> dict[str, str]:
    """
    Export a ZebraEmbedder checkpoint to ONNX.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_dir:      Directory to write ONNX files.
        backbone_name:   'megadescriptor' or 'resnet50'.
        embedding_dim:   Must match the trained model's embedding_dim.
        img_size:        Input image size (384 for MegaDescriptor, 224 for ResNet50).
        opset:           ONNX opset version (17 recommended).
        quantize_int8:   If True, also export a dynamic-range INT8 quantized model.

    Returns:
        Dict with paths to exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")   # export on CPU for portability
    model = build_embedder(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=False,
        device=device,
    )
    raw_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        model.load_state_dict(raw_ckpt["model"])
    else:
        model.load_state_dict(raw_ckpt)
    model.eval()

    # ── Dummy input ───────────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, img_size, img_size)

    # ── Export float32 ONNX ───────────────────────────────────────────────────
    fp32_path = output_dir / f"zebraid_{backbone_name}_fp32.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image":     {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"[export] Saved float32 ONNX → {fp32_path}")

    # ── Validate ONNX output matches PyTorch ──────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        pt_out  = model(dummy).detach().numpy()
        ort_out = sess.run(None, {"image": dummy.numpy()})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f"[export] PyTorch vs ONNX max diff: {max_diff:.6f} (should be < 1e-4)")
        assert max_diff < 1e-4, f"ONNX output mismatch! Max diff: {max_diff}"
    except ImportError:
        print("[export] onnxruntime not available — skipping validation.")

    # ── INT8 quantized ONNX ───────────────────────────────────────────────────
    int8_path = None
    if quantize_int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_path = output_dir / f"zebraid_{backbone_name}_int8.onnx"
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
            print(f"[export] Saved INT8 quantized ONNX → {int8_path}")
            print(
                f"[export] Size: fp32={fp32_path.stat().st_size / 1e6:.1f}MB  "
                f"int8={int8_path.stat().st_size / 1e6:.1f}MB"
            )
        except Exception as e:
            print(f"[export] INT8 quantization failed: {e}")

    exported = {"fp32_onnx": str(fp32_path)}
    if int8_path:
        exported["int8_onnx"] = str(int8_path)
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ZebraEmbedder to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="checkpoints/onnx")
    parser.add_argument("--backbone",   default="megadescriptor")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--img_size",   type=int, default=384)
    parser.add_argument("--opset",      type=int, default=17)
    parser.add_argument("--no_int8",    action="store_true")
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        img_size=args.img_size,
        opset=args.opset,
        quantize_int8=not args.no_int8,
    )
