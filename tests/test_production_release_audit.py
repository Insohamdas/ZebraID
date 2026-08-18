"""
tests/test_production_release_audit.py
Automated release audit test suite for ZebraID v1.0.

Verifies:
  1. Production Checkpoint SHA-256 & byte size
  2. Dataset zero identity leakage
  3. Preprocessing transformation contract
  4. FastAPI security and edge-case handling
  5. Release manifest consistency
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from demo.app import app
from zebraid.data.loaders import build_datasets
from zebraid.data.transforms import eval_transforms
from zebraid.models.backbone import build_embedder

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SHA256 = "3321915956649db169da38b8fa9ca1b26e735bcc40d71306ac83009909e88a80"


def test_production_checkpoint_sha256():
    prod_ckpt = REPO_ROOT / "production" / "model" / "best_model.pt"
    assert prod_ckpt.exists(), "Production checkpoint missing"
    data = prod_ckpt.read_bytes()
    assert len(data) == 818769972
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256


def test_dataset_zero_leakage_audit():
    ds_a_tr, ds_b_tr = build_datasets("train", transform=None, split_seed=42, min_images_per_individual=2)
    ds_a_va, ds_b_va = build_datasets("val", transform=None, split_seed=42, min_images_per_individual=2)
    ds_a_te, ds_b_te = build_datasets("test", transform=None, split_seed=42, min_images_per_individual=2)

    a_tr = set(ds_a_tr.individual_ids)
    a_va = set(ds_a_va.individual_ids)
    a_te = set(ds_a_te.individual_ids)

    b_tr = set(ds_b_tr.individual_ids)
    b_va = set(ds_b_va.individual_ids)
    b_te = set(ds_b_te.individual_ids)

    # Disjointness
    assert len(a_tr & a_va) == 0
    assert len(a_tr & a_te) == 0
    assert len(a_va & a_te) == 0

    assert len(b_tr & b_va) == 0
    assert len(b_tr & b_te) == 0
    assert len(b_va & b_te) == 0

    assert len((a_tr | a_va | a_te) & (b_tr | b_va | b_te)) == 0


def test_preprocessing_contract():
    transform = eval_transforms(384)
    img = Image.new("RGB", (500, 300), color=(100, 150, 200))
    t = transform(img)
    assert t.shape == (3, 384, 384)
    assert t.dtype == torch.float32


def test_fastapi_health_endpoint():
    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ["ok", "healthy"]


def test_fastapi_identify_valid_image():
    client = TestClient(app, raise_server_exceptions=False)
    dummy = Image.new("RGB", (384, 384), color=(128, 128, 128))
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG")
    res = client.post("/identify", files={"file": ("zebra.jpg", buf.getvalue(), "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert "pipeline" in data
    assert "detections" in data


def test_fastapi_identify_invalid_payloads():
    client = TestClient(app, raise_server_exceptions=False)

    # 1. Empty bytes
    res_empty = client.post("/identify", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert res_empty.status_code == 400

    # 2. Corrupt bytes
    res_corrupt = client.post("/identify", files={"file": ("corrupt.jpg", b"INVALID_BYTE_STREAM", "image/jpeg")})
    assert res_corrupt.status_code == 400

    # 3. Missing file
    res_missing = client.post("/identify", files={})
    assert res_missing.status_code == 422

    # 4. Malformed form
    res_malformed = client.post("/identify", files={"wrong_name": ("test.jpg", b"123", "image/jpeg")})
    assert res_malformed.status_code == 422


def test_release_manifest_integrity():
    manifest_path = REPO_ROOT / "release" / "final_release_manifest.json"
    assert manifest_path.exists(), "Release manifest missing"
    with open(manifest_path, "r") as f:
        data = json.load(f)
    assert data["selected_production_checkpoint"]["sha256"] == EXPECTED_SHA256
    assert data["selected_production_checkpoint"]["seed"] == 44
    assert data["selected_production_checkpoint"]["best_epoch"] == 12
