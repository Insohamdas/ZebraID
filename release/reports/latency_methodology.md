# ZebraID Production Latency Methodology & Audit Report

**Hardware Accelerator:** `mps` (Apple Metal Performance Shaders / GPU)  
**Timing Framework:** Python `time.perf_counter()` with explicit hardware barrier synchronization (`torch.mps.synchronize()`)  
**Warm-Up Iterations:** 10  
**Benchmark Sample Size:** 50 iterations  

---

## 1. Asynchronous GPU Timing Audit & Root-Cause Analysis

In earlier asynchronous latency measurements, an artifact was observed where **L2 Normalization** appeared to take ~88 ms while the **Forward Pass** appeared to take ~16 ms.

### Root Cause
PyTorch operations dispatched to Apple MPS or CUDA devices execute asynchronously on the GPU command queue. In un-synchronized benchmarks:
1. `model.forward()` dispatches the vision transformer kernels asynchronously to the GPU and returns immediately to CPU (~16 ms dispatch overhead).
2. The subsequent call to `.cpu()` or `.numpy()` forces the CPU to wait for GPU completion (synchronization barrier).
3. Consequently, the actual heavy vision transformer execution time (~105 ms) was captured in the timer of the *subsequent* stage rather than the forward-pass stage itself.

### Resolution
All latency stages are now strictly bracketed by explicit synchronization barriers (`torch.mps.synchronize()`). Each pipeline component is isolated and measured with true hardware timing.

---

## 2. Synchronized Benchmark Results

| Stage | Description | Mean Latency | Median Latency | p95 Latency | Min Latency | Max Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Preprocessing** | $384 \times 384$ resize + ImageNet normalization (CPU) | **6.89 ms** | 7.0 ms | 8.37 ms | 3.71 ms | 8.73 ms |
| **Forward Pass** | MegaDescriptor-L-384 + 2-layer MLP on GPU | **104.56 ms** | 104.64 ms | 105.19 ms | 102.99 ms | 105.29 ms |
| **L2 Normalization** | Unit hypersphere projection + CPU host copy | **1.27 ms** | 1.3 ms | 2.47 ms | 0.35 ms | 2.76 ms |
| **Gallery Search** | Cosine dot-product ranking over 1,000 individuals | **0.21 ms** | 0.2 ms | 0.26 ms | 0.14 ms | 0.31 ms |
| **Total End-to-End** | Full Image-to-Match Retrieval | **112.93 ms** | **113.19 ms** | **114.86 ms** | 108.27 ms | 116.32 ms |

- **Peak Single-Stream Throughput:** **8.9 images/sec** on mps
