"""
zebraid/models/zhash.py
⭐ NOVEL — ZHashEncoder: compact binary embedding compression.

Compresses a high-dimensional L2-normalized embedding (e.g. 512-d float32)
into a fixed-size binary code — the "Z-Hash" — which is:
  - Transmitted over LoRaWAN from a field station (not the raw image).
  - Stored in the FAISS index for fast approximate matching.
  - The privacy-preserving token in the federated cross-match protocol.

Two backends:
  1. pca_binarize (default): PCA dimensionality reduction → binarize at zero.
     Fast, interpretable, no extra training step.
  2. faiss_pq: FAISS Product Quantization. More accurate at larger scales;
     reuses the same FAISS infrastructure as the matching engine.

Supports benchmarking 128b / 256b / 512b codes (for the accuracy-vs-size
trade-off curve reported in the paper).
"""

from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


Backend = Literal["pca_binarize", "faiss_pq"]


class ZHashEncoder:
    """
    Encodes float32 embeddings into fixed-size binary codes (Z-Hashes).

    Args:
        size_bits:  Target code size in bits. Must be one of 128, 256, 512.
        backend:    'pca_binarize' or 'faiss_pq'.
        n_subquantizers: Number of PQ sub-quantizers (only for faiss_pq backend).
                         Must divide embedding_dim evenly.

    Lifecycle:
        encoder = ZHashEncoder(size_bits=256, backend='pca_binarize')
        encoder.fit(embedding_matrix)  # fit on training embeddings
        code_bytes = encoder.encode(single_embedding)  # 32 bytes for 256b
        approx_emb = encoder.decode(code_bytes)         # approx reconstruction
        encoder.save("checkpoints/zhash_256b.pkl")
        encoder2 = ZHashEncoder.load("checkpoints/zhash_256b.pkl")
    """

    VALID_SIZES = {128, 256, 512}

    def __init__(
        self,
        size_bits: int = 256,
        backend: Backend = "pca_binarize",
        n_subquantizers: int = 16,
    ) -> None:
        if size_bits not in self.VALID_SIZES:
            raise ValueError(f"size_bits must be one of {self.VALID_SIZES}, got {size_bits}")

        self.size_bits = size_bits
        self.size_bytes = size_bits // 8
        self.backend = backend
        self.n_subquantizers = n_subquantizers
        self._is_fitted = False

        # Internal state (set during fit)
        self._pca: Optional[object] = None
        self._pq_index: Optional[object] = None

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> "ZHashEncoder":
        """
        Fit the encoder on a matrix of L2-normalized embeddings.

        Args:
            embeddings: shape (N, D), float32, L2-normalized.
        Returns:
            self (for chaining)
        """
        embeddings = embeddings.astype(np.float32)
        N, D = embeddings.shape

        if self.backend == "pca_binarize":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for pca_binarize backend.")
            n_components = self.size_bits  # 1 PCA component → 1 bit
            self._pca = PCA(n_components=min(n_components, D, N))
            self._pca.fit(embeddings)

        elif self.backend == "faiss_pq":
            if not FAISS_AVAILABLE:
                raise ImportError("faiss is required for faiss_pq backend.")
            # FAISS PQ: size_bits / 8 bytes, n_subquantizers centroids per sub-space
            # bits_per_subquantizer = size_bits // n_subquantizers (must be 8 for byte-aligned)
            if self.size_bits % self.n_subquantizers != 0:
                raise ValueError(
                    f"size_bits ({self.size_bits}) must be divisible by "
                    f"n_subquantizers ({self.n_subquantizers})"
                )
            bits_per_sub = self.size_bits // self.n_subquantizers
            self._pq_index = faiss.IndexPQ(D, self.n_subquantizers, bits_per_sub)
            self._pq_index.train(embeddings)

        self._embedding_dim = D
        self._is_fitted = True
        return self

    # ── Encode / Decode ────────────────────────────────────────────────────

    def encode(self, embedding: np.ndarray) -> bytes:
        """
        Encode a single L2-normalized embedding into a binary Z-Hash.

        Args:
            embedding: shape (D,) or (1, D), float32.
        Returns:
            bytes of length self.size_bytes (e.g. 32 bytes for 256 bits).
        """
        self._check_fitted()
        embedding = embedding.astype(np.float32).reshape(1, -1)

        if self.backend == "pca_binarize":
            reduced = self._pca.transform(embedding)[0]  # (n_components,)
            # Pad/truncate to exactly size_bits bits
            bits = np.zeros(self.size_bits, dtype=np.uint8)
            n = min(self.size_bits, len(reduced))
            bits[:n] = (reduced[:n] > 0).astype(np.uint8)
            return np.packbits(bits).tobytes()  # exactly size_bytes bytes

        elif self.backend == "faiss_pq":
            codes = self._pq_index.sa_encode(embedding)  # (1, size_bytes)
            return codes[0].tobytes()

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a batch of embeddings.

        Args:
            embeddings: shape (N, D), float32.
        Returns:
            np.ndarray of dtype uint8, shape (N, size_bytes).
        """
        self._check_fitted()
        embeddings = embeddings.astype(np.float32)

        if self.backend == "pca_binarize":
            reduced = self._pca.transform(embeddings)  # (N, n_components)
            # Take only the first size_bits components (PCA may have fewer if D<size_bits)
            n_bits_actual = min(self.size_bits, reduced.shape[1])
            bits = np.zeros((len(embeddings), self.size_bits), dtype=np.uint8)
            bits[:, :n_bits_actual] = (reduced[:, :n_bits_actual] > 0).astype(np.uint8)
            # Pack 8 bits per byte → exactly size_bytes bytes per row
            packed = np.packbits(bits, axis=1)  # (N, size_bytes)
            return packed

        elif self.backend == "faiss_pq":
            return self._pq_index.sa_encode(embeddings)  # (N, size_bytes)

    def decode(self, code: bytes) -> np.ndarray:
        """
        Approximate reconstruction of an embedding from its Z-Hash.
        Used for validation — not required in production matching.

        Returns: shape (D,), float32.
        """
        self._check_fitted()
        code_array = np.frombuffer(code, dtype=np.uint8).reshape(1, -1)

        if self.backend == "pca_binarize":
            bits = np.unpackbits(code_array, axis=1)[0, : self.size_bits]
            # Map bits back to ±1 — only use the actual n_components the PCA has
            signed = (bits.astype(np.float32) * 2) - 1
            n_comp = self._pca.n_components_  # actual number of components fitted
            return self._pca.inverse_transform(signed[:n_comp].reshape(1, -1))[0]

        elif self.backend == "faiss_pq":
            decoded = self._pq_index.sa_decode(code_array)  # (1, D)
            return decoded[0]

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save encoder state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "size_bits": self.size_bits,
            "backend": self.backend,
            "n_subquantizers": self.n_subquantizers,
            "_embedding_dim": getattr(self, "_embedding_dim", None),
            "_pca": self._pca,
        }
        if self.backend == "faiss_pq" and self._pq_index is not None:
            import faiss
            faiss.write_index(self._pq_index, str(path) + ".pq.faiss")
            state["_pq_index"] = None  # stored separately

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "ZHashEncoder":
        """Load encoder from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        enc = cls(
            size_bits=state["size_bits"],
            backend=state["backend"],
            n_subquantizers=state["n_subquantizers"],
        )
        enc._pca = state["_pca"]
        enc._embedding_dim = state.get("_embedding_dim")

        if enc.backend == "faiss_pq":
            import faiss
            enc._pq_index = faiss.read_index(str(path) + ".pq.faiss")

        enc._is_fitted = True
        return enc

    # ── Payload info ─────────────────────────────────────────────────────────

    @property
    def payload_size_bytes(self) -> int:
        """Size of the Z-Hash payload transmitted over LoRaWAN."""
        return self.size_bytes

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ZHashEncoder must be fitted before encoding. Call .fit(embeddings).")
