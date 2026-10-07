"""Hybrid retrieval with a relevance gate.

1. BM25 ranks the chunks, and the embedder ranks the chunks by cosine.
2. Reciprocal-rank fusion (RRF, k = 60) gives one list.
3. The relevance gate: the best passage must contain at least ``min_coverage`` of the content
   tokens of the query. If no passage passes, the assistant says that it has no source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .index import KnowledgeIndex
from .ingest import Chunk
from .text import content_tokens, query_tokens, sentences

RRF_K = 60


@dataclass
class Passage:
    number: int  # 1-based, as shown in the prompt and the citations
    chunk: Chunk
    bm25: float
    cosine: float
    coverage: float
    fused: float


@dataclass
class Retrieval:
    passages: list[Passage]
    has_evidence: bool
    best_coverage: float


def retrieve(idx: KnowledgeIndex, embedder: Embedder, query: str, k: int, min_coverage: float,
             pool: int = 30) -> Retrieval:
    q_tokens = set(query_tokens(query))
    if not q_tokens or not idx.chunks:
        return Retrieval([], False, 0.0)
    # Coverage is measured against the whole query and against each question sentence with two or
    # more content tokens, so a long preamble ("I am pregnant. ...") does not hide a clear question.
    q_sets = [set(content_tokens(query))] + [
        t for t in (set(content_tokens(x)) for x in sentences(query)) if len(t) >= 2
    ]
    q_sets = [q for q in q_sets if q]
    bm = idx.bm25.scores(list(q_tokens))
    cos = idx.vectors @ embedder.embed([query])[0]
    pool = min(pool, len(idx.chunks))
    rank_bm = np.argsort(-bm, kind="stable")[:pool]
    rank_cos = np.argsort(-cos, kind="stable")[:pool]
    fused: dict[int, float] = {}
    for ranking in (rank_bm, rank_cos):
        for r, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
    order = sorted(fused, key=lambda i: (-fused[i], i))
    out, seen = [], set()
    for i in order:
        c = idx.chunks[i]
        key = (c.source, c.section, c.text[:80])
        if key in seen:
            continue
        seen.add(key)
        cov = max((len(q & idx.token_sets[i]) / len(q) for q in q_sets), default=0.0)
        out.append(Passage(0, c, float(bm[i]), float(cos[i]), cov, fused[i]))
        if len(out) == k:
            break
    out = [p for p in out if p.coverage > 0]  # a passage with no query token is never context
    for n, p in enumerate(out, 1):
        p.number = n
    best = max((p.coverage for p in out), default=0.0)
    return Retrieval(out, best >= min_coverage, best)
