# ZebraID — Final Production Inference Validation Report

**Deployment State:** Immutable Research & Production Freeze (v1.0)  
**Validated Checkpoint:** `production/model/best_model.pt`  
**Cryptographic Integrity:** `SHA-256: 3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80`  
**Execution Decision:** **`READY FOR PRODUCTION`** ✅  

---

## 1. Model Loading & Architecture Verification

- **Backbone Architecture:** MegaDescriptor-L-384 (`hf-hub:BVRA/MegaDescriptor-L-384`)
- **Projector Architecture:** 2-Layer MLP ($2048 \rightarrow 2048 \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow 512 \rightarrow \text{L2 Normalize}$)
- **Total Parameters:** 198,349,364
- **Trainable Parameters:** 0 (Gradients strictly disabled during inference)
- **Frozen Parameters:** 198,349,364
- **Embedding Dimension:** `512`
- **Weight Integrity:** 100% finite parameters (0 NaNs / 0 Infs).

---

## 2. Preprocessing & Embedding Assertions

Tested across 25 real wildlife images:
- **Embedding Shape:** $(1, 512)$ for 100% of samples
- **L2 Unit Norm:** $\text{Mean} = 1.000000$, $\text{Min} = 1.000000$, $\text{Max} = 1.000000$
- **NaN Embeddings:** `0`
- **Inf Embeddings:** `0`

---

## 3. Latency & Batch Throughput Profiling

Measured on `mps` with explicit hardware synchronization (`torch.mps.synchronize()`):

| Inference Stage | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) |
|---|:---:|:---:|:---:|
| **1. Image Preprocessing** | 6.89 ms | 7.00 ms | 8.37 ms |
| **2. Backbone & Projector Forward** | 104.56 ms | 104.64 ms | 105.19 ms |
| **3. L2 Normalization & Host Copy** | 1.27 ms | 1.30 ms | 2.47 ms |
| **4. Gallery Retrieval (1k Gallery)** | 0.21 ms | 0.20 ms | 0.26 ms |
| **Total End-to-End Pipeline** | **112.93 ms** | **113.19 ms** | **114.86 ms** |

- **Peak Single-Stream Throughput:** **8.9 images/sec** on Apple Silicon MPS

### Batch Scaling
- **Batch 1:** 85.69 ms/image (11.7 img/sec)
- **Batch 8:** 103.04 ms/image (9.7 img/sec)
- **Batch 16:** 104.42 ms/image (9.6 img/sec)

---

## 4. API & Safety Edge-Case Handling

- **`GET /health`:** `200 OK`
- **`POST /identify` (Valid):** `200 OK` (Generated 512-d embedding & 256-bit Z-Hash)
- **`POST /identify` (Empty / Corrupted Payload):** `400 Bad Request` (Safely caught without server crash)
- **`POST /identify` (Missing File):** `422 Unprocessable Entity`
- **`POST /identify` (Malformed Form):** `422 Unprocessable Entity`
