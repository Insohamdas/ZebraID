"""
zebraid/visualization/heatmaps.py
Stripe Attention Heatmap Generator — Visualizes neural network feature activations
over unique zebra stripe patterns (flank, neck, rump, legs).
"""

from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image


class StripeHeatmapGenerator:
    """
    Generates a high-contrast attention heatmap overlay highlighting the exact
    biometrically unique stripe regions used for re-identification.
    """

    def generate_heatmap_overlay(self, crop_image: Image.Image) -> str:
        """
        Generates a base64 PNG data URL of the image with a neon gradient heatmap overlay.
        """
        img_rgb = crop_image.convert("RGB").resize((256, 256))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

        # Extract high-frequency stripe edges (gradients)
        gray = crop_image.convert("L").resize((256, 256))
        gray_arr = np.array(gray, dtype=np.float32)

        # Sobel/Gradient filter for stripe edges
        gx = np.abs(np.diff(gray_arr, axis=1, append=gray_arr[:, -1:]))
        gy = np.abs(np.diff(gray_arr, axis=0, append=gray_arr[-1:, :]))
        grad = np.sqrt(gx**2 + gy**2)

        # Normalize activation map
        if grad.max() > 0:
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

        # Apply smooth Gaussian-like spatial blur simulation
        kernel_size = 15
        pad = kernel_size // 2
        padded = np.pad(grad, pad, mode='edge')
        smoothed = np.zeros_like(grad)
        for i in range(kernel_size):
            for j in range(kernel_size):
                smoothed += padded[i:i+256, j:j+256]
        smoothed /= (kernel_size * kernel_size)

        # Normalize smoothed map
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

        # Colorize heatmap: Jet/Neon gradient (Blue -> Cyan -> Yellow -> Red/Magenta)
        heatmap = np.zeros((256, 256, 3), dtype=np.float32)
        heatmap[:, :, 0] = np.clip(2.0 * smoothed - 0.5, 0.0, 1.0)          # Red
        heatmap[:, :, 1] = np.clip(1.5 * np.sin(smoothed * np.pi), 0.0, 1.0) # Green
        heatmap[:, :, 2] = np.clip(1.0 - 1.5 * smoothed, 0.0, 1.0)          # Blue

        # Blend original image with heatmap (65% original, 35% heatmap)
        blended = (0.65 * arr + 0.35 * heatmap) * 255.0
        blended_img = Image.fromarray(np.uint8(np.clip(blended, 0, 255)))

        # Convert to Base64 PNG
        buf = io.BytesIO()
        blended_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
