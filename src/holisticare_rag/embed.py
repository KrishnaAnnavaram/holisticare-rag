"""Embedders behind one interface. Each gives unit-length float32 vectors.

* ``HashingEmbedder`` (default, offline): hashed word unigrams and bigrams of content tokens.
* ``SentenceTransformerEmbedder`` (extra ``st``): a local sentence-transformers model.
* ``OllamaEmbedder``: the local Ollama server (``/api/embed``), standard library HTTP only.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .text import content_tokens


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing-1024"

    def __init__(self, n_features: int = 1024):
        self.vec = HashingVectorizer(n_features=n_features, analyzer=lambda t: self._grams(t),
                                     alternate_sign=False, norm=None)

    @staticmethod
    def _grams(text: str) -> list[str]:
        toks = content_tokens(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts):
        m = self.vec.transform(texts).toarray()
        return _unit(np.log1p(m))


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415  (optional extra)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('this embedder needs: pip install "holisticare-rag[st]"') from exc
        self.model = SentenceTransformer(model, device="cpu")
        self.name = f"st:{model}"

    def embed(self, texts):  # pragma: no cover - heavy model
        return _unit(np.asarray(self.model.encode(texts, batch_size=32, show_progress_bar=False)))


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str | None = None, timeout_s: float = 60.0):
        self.url = (base_url or "http://localhost:11434").rstrip("/") + "/api/embed"
        self.model, self.timeout = model, timeout_s
        self.name = f"ollama:{model}"

    def embed(self, texts):  # pragma: no cover - network
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return _unit(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))


def make_embedder(kind: str, model: str, base_url: str | None = None, timeout_s: float = 60.0) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    if kind == "ollama":
        return OllamaEmbedder(model, base_url, timeout_s)
    raise ValueError(f"unknown embedder {kind!r}")
