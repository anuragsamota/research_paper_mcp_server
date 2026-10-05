"""Pluggable text embedders.

* ``sentence-transformers`` – neural embeddings run locally (best quality; ``pip install .[st]``).
* ``ollama`` – embeddings from an Ollama server (e.g. ``nomic-embed-text``).
* ``hash`` – dependency-free hashed TF n-gram vectors. Works offline everywhere; lexical
  rather than truly semantic, so the retriever blends it with BM25.

``auto`` picks sentence-transformers if installed, otherwise ``hash``.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from .config import Settings
from .text import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an L2-normalized float32 matrix of shape (len(texts), dim)."""
        ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Signed feature hashing of unigrams + bigrams with sublinear term frequency."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text, stemmed=True)
            features = Counter(tokens)
            features.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
            for feature, count in features.items():
                idx, sign = self._bucket(feature)
                weight = 1.0 + math.log(count)
                if "_" in feature:
                    weight *= 0.7
                out[row, idx] += sign * weight
        return _normalize_rows(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = ""):
        from sentence_transformers import SentenceTransformer  # optional dependency

        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return _normalize_rows(vectors)


class OllamaEmbedder:
    def __init__(self, url: str, model: str = "", timeout: float = 60.0, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.model = model or "nomic-embed-text"
        self._client = client or httpx.Client(timeout=timeout)
        self.name = f"ollama:{self.model}"
        self.dim = len(self._request(["dimension probe"])[0])

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            vectors.extend(self._request(texts[start : start + 64]))
        return _normalize_rows(np.array(vectors))


def create_embedder(settings: Settings) -> Embedder:
    choice = settings.embedder.lower()
    if choice in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except ImportError:
            if choice != "auto":
                raise RuntimeError("sentence-transformers is not installed: pip install 'research-paper-mcp[st]'")
            log.info("sentence-transformers not installed; using the hashing embedder")
        except Exception as exc:  # e.g. model download blocked
            if choice != "auto":
                raise
            log.warning("Could not load sentence-transformers model (%s); using the hashing embedder", exc)
        return HashingEmbedder()
    if choice == "ollama":
        return OllamaEmbedder(settings.ollama_url, settings.embedding_model, settings.http_timeout)
    if choice == "hash":
        return HashingEmbedder()
    raise ValueError(f"Unknown RESEARCH_EMBEDDER {settings.embedder!r} (use auto, hash, sentence-transformers or ollama)")
