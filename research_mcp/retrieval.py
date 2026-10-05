"""Hybrid retrieval: dense vector search + BM25, fused with reciprocal rank fusion (RRF)."""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .store import Store, StoredChunk
from .text import tokenize


@dataclass
class Hit:
    chunk: StoredChunk
    score: float  # fused score (higher is better)
    dense: float  # cosine similarity
    lexical: float  # BM25 score


class _Bm25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        if not self.avgdl:
            return out
        terms = [t for t in set(query) if t in self.idf]
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


class VectorIndex:
    """In-memory index over every chunk in the store, rebuilt lazily when the store changes."""

    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._lock = threading.Lock()
        self._version = -1
        self._chunks: list[StoredChunk] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25 = _Bm25([])

    def _refresh(self) -> None:
        with self._lock:
            if self._version == self.store.version:
                return
            chunks, vectors = [], []
            for chunk, vector in self.store.iter_chunks():
                if vector is None or vector.shape[0] != self.embedder.dim:
                    continue  # embedded by another model; `reindex` fixes this
                chunks.append(chunk)
                vectors.append(vector)
            self._chunks = chunks
            self._matrix = np.vstack(vectors) if vectors else np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._bm25 = _Bm25([tokenize(c.text, stemmed=True) for c in chunks])
            self._version = self.store.version

    @property
    def size(self) -> int:
        self._refresh()
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 8,
        paper_ids: list[str] | None = None,
        sections: list[str] | None = None,
        per_paper_limit: int | None = None,
    ) -> list[Hit]:
        self._refresh()
        if not self._chunks or not query.strip():
            return []
        mask = np.ones(len(self._chunks), dtype=bool)
        if paper_ids:
            wanted = set(paper_ids)
            mask &= np.array([c.paper_id in wanted for c in self._chunks])
        if sections:
            wanted_sections = set(sections)
            mask &= np.array([c.section in wanted_sections for c in self._chunks])
        if not mask.any():
            return []

        qvec = self.embedder.embed([query])[0]
        dense = self._matrix @ qvec
        lexical = self._bm25.scores(tokenize(query, stemmed=True))

        candidates = np.flatnonzero(mask)
        dense_rank = _ranks(dense, candidates)
        lex_rank = _ranks(lexical, candidates, require_positive=True)
        k = 60.0
        fused = {i: 1.0 / (k + dense_rank[i]) + (1.0 / (k + lex_rank[i]) if i in lex_rank else 0.0) for i in candidates}

        hits: list[Hit] = []
        per_paper: Counter[str] = Counter()
        for i in sorted(fused, key=fused.get, reverse=True):
            chunk = self._chunks[i]
            if per_paper_limit and per_paper[chunk.paper_id] >= per_paper_limit:
                continue
            per_paper[chunk.paper_id] += 1
            hits.append(Hit(chunk, float(fused[i]), float(dense[i]), float(lexical[i])))
            if len(hits) >= top_k:
                break
        return hits

    def paper_vectors(self, paper_ids: list[str] | None = None) -> dict[str, np.ndarray]:
        """Mean (re-normalized) chunk embedding per paper – a document-level representation."""
        self._refresh()
        groups: dict[str, list[int]] = {}
        for i, chunk in enumerate(self._chunks):
            if paper_ids is None or chunk.paper_id in paper_ids:
                groups.setdefault(chunk.paper_id, []).append(i)
        out = {}
        for pid, rows in groups.items():
            v = self._matrix[rows].mean(axis=0)
            norm = np.linalg.norm(v)
            out[pid] = v / norm if norm else v
        return out


def _ranks(scores: np.ndarray, candidates: np.ndarray, require_positive: bool = False) -> dict[int, int]:
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    if require_positive:
        order = [i for i in order if scores[i] > 0]
    return {int(i): rank for rank, i in enumerate(order, start=1)}
