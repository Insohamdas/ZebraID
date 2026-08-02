"""tests/test_zhash.py — Component 7"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from zebraid.models.zhash import ZHashEncoder


def _random_embeddings(n=200, d=512, seed=42):
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


@pytest.mark.parametrize("size_bits", [128, 256, 512])
def test_encode_size_is_correct(size_bits):
    embs = _random_embeddings()
    enc = ZHashEncoder(size_bits=size_bits, backend="pca_binarize")
    enc.fit(embs)
    code = enc.encode(embs[0])
    assert isinstance(code, bytes)
    assert len(code) == size_bits // 8, f"Expected {size_bits//8} bytes, got {len(code)}"


@pytest.mark.parametrize("size_bits", [128, 256, 512])
def test_batch_encode_shape(size_bits):
    embs = _random_embeddings()
    enc = ZHashEncoder(size_bits=size_bits, backend="pca_binarize")
    enc.fit(embs)
    codes = enc.encode_batch(embs[:10])
    assert codes.shape == (10, size_bits // 8)


def test_encode_decode_roundtrip_pca():
    """PCA decode should produce an approximation (not exact but reasonable)."""
    embs = _random_embeddings()
    enc = ZHashEncoder(size_bits=256, backend="pca_binarize")
    enc.fit(embs)
    code = enc.encode(embs[0])
    approx = enc.decode(code)
    assert approx.shape == (512,)
    # Cosine similarity with original should be > 0 (positive correlation)
    cos_sim = float(np.dot(embs[0], approx) / (np.linalg.norm(embs[0]) * np.linalg.norm(approx) + 1e-8))
    assert cos_sim > 0, f"Decoded embedding has negative correlation: {cos_sim}"


def test_save_and_load(tmp_path):
    embs = _random_embeddings()
    enc = ZHashEncoder(size_bits=256, backend="pca_binarize")
    enc.fit(embs)
    save_path = tmp_path / "test_encoder.pkl"
    enc.save(save_path)

    enc2 = ZHashEncoder.load(save_path)
    code1 = enc.encode(embs[5])
    code2 = enc2.encode(embs[5])
    assert code1 == code2, "Loaded encoder produces different codes"


def test_invalid_size_raises():
    with pytest.raises(ValueError):
        ZHashEncoder(size_bits=64)  # not in {128, 256, 512}


def test_payload_size_bytes_property():
    enc = ZHashEncoder(size_bits=256)
    assert enc.payload_size_bytes == 32

    enc2 = ZHashEncoder(size_bits=128)
    assert enc2.payload_size_bytes == 16
