# ZebraID

**Stripe-based biometric identification system for individual zebra recognition at continental scale.**

> ⭐ Novel contributions: cross-population embedding, Z-Hash compression, federated cross-match protocol.

## Quick Start (Mac mini)

```bash
# 1. Clone and install
git clone <your-repo>
cd zebraid
pip install -r requirements.txt
pip install -e .

# 2. Download datasets
bash data/download_gzgc.sh
bash data/download_second_pop.sh

# 3. Run tests (no GPU needed)
pytest tests/ -v

# 4. Start the federated demo (UI at http://localhost:8000)
# Terminal 1 — Org A shard:
ORG_ID=OrgA uvicorn zebraid.federation.org_service:create_app \
    --factory --port 8001

# Terminal 2 — Org B shard:
ORG_ID=OrgB uvicorn zebraid.federation.org_service:create_app \
    --factory --port 8002

# Terminal 3 — Coordinator UI:
python demo/app.py
```

## Training (Google Colab GPU)

Upload this repo to Colab and run `notebooks/02_training.ipynb`.  
The three training modes produce the paper's core comparison table:

| Mode | Trained on | Evaluated on | Purpose |
|---|---|---|---|
| `baseline_a` | Pop A | Pop A | Single-population baseline |
| `baseline_x` | Pop A | Pop B | Generalization gap measurement |
| `zebraid` | Mixed A+B | Both | ⭐ Cross-population result |

## Project Structure

```
zebraid/
├── data/          dataset loaders, transforms, mixed batch sampler
├── models/        backbone, loss, training, Z-Hash encoder
├── matching/      FAISS index, scale simulation
├── federation/    OrgShard FastAPI service, federation client, privacy audit
├── edge/          ONNX export, CPU-proxy benchmark
configs/           default.yaml
demo/              coordinator app + web UI
notebooks/         experiment notebooks (run on Colab)
tests/             unit + integration tests
```

## Hardware Note

All software development and testing runs on **Mac mini (CPU)**. Physical edge hardware is to be collected:
- **Sensor node**: ESP32-CAM + LoRa module
- **Inference node**: Raspberry Pi 5 or Jetson Orin Nano

The ONNX model exported by `zebraid/edge/export.py` runs unchanged on all platforms.

## License

MIT — See LICENSE.  
Dataset attribution: GZGC (Parham et al., 2017) under CDLA-Permissive-1.0.
