# ZebraID Model Card & Specification

## Model Details
- **Model Name**: ZebraID Dual-Population Stripe-Biometric Embedder
- **Architecture**: MegaDescriptor (Swin-Base / ConvNeXt backbone) + Z-Hash 256-bit Binary Quantization
- **Primary Domain**: Non-invasive individual re-identification of Equids (*Equus quagga* plains zebra and *Equus grevyi* Grevy's zebra)
- **Framework**: PyTorch / ONNX Runtime / FAISS-CPU

## Intended Use
- **Primary Use**: Continental-scale wildlife monitoring, non-invasive demographic tracking, anti-poaching patrol search, cross-park population census.
- **Privacy Guarantees**: Query responses **never transmit raw imagery or raw GPS coordinates** across organization boundaries. Score outputs are quantized into coarse confidence buckets (`STRONG_MATCH`, `WEAK_MATCH`, `NO_MATCH`).

## Datasets & Training
- **Population A**: Great Zebra & Giraffe Count (GZGC) — 1,905 individuals, 6,286 annotations.
- **Population B**: Labeled Mpala Grevy's Zebra — 173 individuals, 685 annotations.
- **Sampling Strategy**: `MixedPopulationBatchSampler` ensuring proportional representation of both species in every mini-batch during cross-population metric learning.

## Vector Compression (Z-Hash)
- **Full Embedding**: 512-dimensional FP32 (2,048 bytes per individual)
- **Deployed Z-Hash**: 256-bit packed binary payload (32 bytes per individual)
- **Compression Ratio**: 64x memory savings vs raw float32 embeddings

## Hardware Targets
- **Cloud / Federation Node**: Mac mini / Linux Server (FAISS IVF-PQ + SQLite Audit Log)
- **Edge Deployment**: Raspberry Pi 5 / NVIDIA Jetson Orin Nano / ESP32-CAM (CPU-Proxy simulation verified via ONNX Runtime INT8)
