"""The knowledge index: chunks, BM25 statistics and vectors, versioned by a content fingerprint.

The index is plain JSON, JSON Lines and a NumPy array. Nothing is unpickled. ``ensure_index``
builds the index again when the documents, the registry, the chunk settings or the embedder change.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .embed import Embedder
from .ingest import Chunk, chunk_to_dict, docs_fingerprint, ingest
from .text import content_tokens

FORMAT_VERSION = 1


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in set(query):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out[i] = s
        return out


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict

    def __post_init__(self) -> None:
        self.tokens = [content_tokens(f"{c.title} {c.section} {c.text}") for c in self.chunks]
        self.token_sets = [set(t) for t in self.tokens]
        self.bm25 = BM25(self.tokens)


def settings_key(s: Settings, embedder: Embedder) -> str:
    return f"v{FORMAT_VERSION}|{s.chunk_chars}|{s.chunk_overlap}|{embedder.name}"


def build_index(s: Settings, embedder: Embedder) -> tuple[KnowledgeIndex, list[str]]:
    chunks, rep = ingest(s.docs_dir, s.chunk_chars, s.chunk_overlap)
    texts = [f"{c.title}. {c.section}. {c.text}" for c in chunks]
    vectors = embedder.embed(texts)
    manifest = {
        "format_version": FORMAT_VERSION,
        "fingerprint": docs_fingerprint(s.docs_dir, settings_key(s, embedder)),
        "embedder": embedder.name, "documents": rep.documents, "chunks": rep.chunks,
        "chunk_chars": s.chunk_chars, "chunk_overlap": s.chunk_overlap,
    }
    return KnowledgeIndex(chunks, vectors, manifest), rep.warnings


def save_index(idx: KnowledgeIndex, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in idx.chunks:
            fh.write(json.dumps(chunk_to_dict(c)) + "\n")
    np.save(folder / "vectors.npy", idx.vectors, allow_pickle=False)
    (folder / "manifest.json").write_text(json.dumps(idx.manifest, indent=2), encoding="utf-8")


def load_index(folder: Path) -> KnowledgeIndex:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with open(folder / "chunks.jsonl", encoding="utf-8") as fh:
        chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    return KnowledgeIndex(chunks, vectors, manifest)


def ensure_index(s: Settings, embedder: Embedder, force: bool = False) -> tuple[KnowledgeIndex, str]:
    """Load the saved index if its fingerprint matches the documents. Otherwise build and save it.

    Returns the index and a status: ``reused``, ``built`` or ``rebuilt``.
    """
    folder = Path(s.index_dir)
    fp = docs_fingerprint(s.docs_dir, settings_key(s, embedder))
    mpath = folder / "manifest.json"
    if mpath.exists() and not force:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("fingerprint") == fp and m.get("format_version") == FORMAT_VERSION:
            return load_index(folder), "reused"
    status = "rebuilt" if mpath.exists() else "built"
    idx, _ = build_index(s, embedder)
    save_index(idx, folder)
    return idx, status
