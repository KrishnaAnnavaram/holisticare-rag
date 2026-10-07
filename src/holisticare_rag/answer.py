"""Answer construction and checks: extractive answers, citation checks and supporting quotes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieve import Passage
from .text import content_tokens, sentences

TRADITIONAL_NOTE = "Traditional uses in these sources are not established by clinical evidence."
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    number: int
    source: str
    title: str
    page: int | None
    section: str
    evidence: str
    quote: str


@dataclass
class Answer:
    text: str
    category: str  # ok, no_source, emergency, self_harm, dosing, stop_medication
    citations: list[Citation] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    retrieved: list[str] = field(default_factory=list)  # source files of the passages, in rank order
    model: str = ""
    disclaimer: str = ""


def _overlap(a: set[str], b: str) -> int:
    return len(a & set(content_tokens(b)))


def best_sentence(passage: Passage, claim: str) -> str:
    q = set(content_tokens(claim))
    sents = sentences(passage.chunk.text) or [passage.chunk.text]
    return max(sents, key=lambda s: (_overlap(q, s), -len(s)))


def extractive_answer(question: str, passages: list[Passage], max_sentences: int = 4) -> str:
    """One or two best sentences per passage, grouped by evidence label, each with its citation."""
    q = set(content_tokens(question))
    picked: list[tuple[str, str, int]] = []  # (evidence, sentence, number)
    for p in passages:
        ranked = sorted(sentences(p.chunk.text), key=lambda s: (-_overlap(q, s), len(s)))
        for s in ranked[:2]:
            if _overlap(q, s) == 0 or any(s == x[1] for x in picked):
                continue
            picked.append((p.chunk.evidence, s, p.number))
            break
        if len(picked) >= max_sentences:
            break
    groups = {"evidence_based": [], "unknown": [], "traditional": []}
    for ev, s, n in picked:
        body = s.rstrip()
        end = body[-1] if body[-1:] in ".!?" else "."
        groups.setdefault(ev, []).append(f"{body.rstrip('.!?')} [{n}]{end}")
    parts = []
    if groups["evidence_based"]:
        parts.append("From evidence-based sources: " + " ".join(groups["evidence_based"]))
    if groups["unknown"]:
        parts.append("From sources without an evidence label: " + " ".join(groups["unknown"]))
    if groups["traditional"]:
        parts.append("From traditional-medicine sources: " + " ".join(groups["traditional"]) + " " + TRADITIONAL_NOTE)
    return "\n\n".join(parts)


def check_citations(text: str, n_passages: int, min_cited: float = 0.8) -> list[str]:
    """Reasons to refuse a model answer. An empty list means that the answer passes."""
    reasons = []
    sents = [s for s in sentences(text) if len(content_tokens(s)) >= 3]
    if not sents:
        return ["empty answer"]
    nums = [int(n) for n in _CITE.findall(text)]
    bad = sorted({n for n in nums if not 1 <= n <= n_passages})
    if bad:
        reasons.append(f"citations to passages that do not exist: {bad}")
    cited = sum(1 for s in sents if _CITE.search(s))
    if cited / len(sents) < min_cited:
        reasons.append(f"only {cited} of {len(sents)} sentences have a citation")
    return reasons


def citations_for(text: str, passages: list[Passage]) -> list[Citation]:
    """One citation per cited passage, with the passage sentence that best supports the citing sentences."""
    by_num = {p.number: p for p in passages}
    claims: dict[int, list[str]] = {}
    for s in sentences(text):
        for n in {int(x) for x in _CITE.findall(s)}:
            claims.setdefault(n, []).append(_CITE.sub("", s))
    out = []
    for n in sorted(claims):
        p = by_num.get(n)
        if p is None:
            continue
        c = p.chunk
        out.append(Citation(n, c.source, c.title, c.page, c.section, c.evidence, best_sentence(p, " ".join(claims[n]))))
    return out
