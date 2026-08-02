# ZebraID — Makefile
# Usage: make <target>
# Always run from the project root.

# Prefer python3.11 on PATH; override with `make PYTHON=/path/to/python3.11` if needed.
PYTHON     ?= python3.11
VENV       = .venv
VENV_PY    = $(VENV)/bin/python
VENV_PIP   = $(VENV)/bin/pip
TORCH_IDX  = https://download.pytorch.org/whl/cpu

.PHONY: venv install test test-data test-federation clean demo help

# ── Setup ─────────────────────────────────────────────────────────────────────

venv:
	@echo "Creating virtual environment with $(PYTHON)..."
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	@echo "✅  Virtual environment ready: source .venv/bin/activate"

install: venv
	@echo "Installing dependencies..."
	$(VENV_PIP) install -q torch torchvision --index-url $(TORCH_IDX)
	$(VENV_PIP) install -q faiss-cpu scikit-learn timm \
	    pytorch-metric-learning \
	    fastapi uvicorn httpx pydantic python-multipart \
	    pillow pyyaml pandas \
	    pytest pytest-asyncio anyio \
	    onnxruntime wandb
	$(VENV_PIP) install -e .
	@echo "✅  All packages installed"

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	@echo "Running dataset + Z-Hash + real loader tests..."
	$(VENV_PY) -m pytest tests/test_dataset.py tests/test_zhash.py tests/test_loaders.py -v
	@echo "Running privacy + federation tests..."
	$(VENV_PY) -m pytest tests/test_privacy.py tests/test_federation.py -v

test-data:
	$(VENV_PY) -m pytest tests/test_dataset.py tests/test_zhash.py tests/test_loaders.py -v

test-federation:
	$(VENV_PY) -m pytest tests/test_privacy.py tests/test_federation.py -v

test-extra:
	$(VENV_PY) -m pytest tests/test_extra_features.py -v

# ── Demo ──────────────────────────────────────────────────────────────────────

demo:
	@echo "Starting ZebraID demo coordinator on http://localhost:8000"
	@echo "Make sure Org A (port 8001) and Org B (port 8002) shards are running."
	$(VENV_PY) demo/app.py

shard-a:
	ORG_ID=OrgA $(VENV)/bin/uvicorn zebraid.federation.org_service:create_app \
	    --factory --port 8001 --reload

shard-b:
	ORG_ID=OrgB $(VENV)/bin/uvicorn zebraid.federation.org_service:create_app \
	    --factory --port 8002 --reload

# ── Training (Mac mini M2 Pro) ────────────────────────────────────────────────

train-mac:
	@echo "Starting full 30-epoch ZebraID training suite on Apple Silicon (MPS)..."
	caffeinate -i $(VENV_PY) scripts/run_mac_training.py --epochs 30

train-mac-quick:
	@echo "Starting quick 3-epoch test run on Apple Silicon (MPS)..."
	caffeinate -i $(VENV_PY) scripts/run_mac_training.py --epochs 3

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	@echo "✅  Cleaned"


# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "ZebraID — Available make targets:"
	@echo "  make venv          Create Python 3.11 virtual environment"
	@echo "  make install       Create venv + install all dependencies"
	@echo "  make test          Run full test suite"
	@echo "  make test-data     Run dataset + Z-Hash + real loader tests only"
	@echo "  make test-fed      Run privacy + federation tests only"
	@echo "  make demo          Start the coordinator web UI"
	@echo "  make shard-a       Start Org A shard (port 8001)"
	@echo "  make shard-b       Start Org B shard (port 8002)"
	@echo "  make clean         Delete venv and cache files"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make test"
	@echo ""
