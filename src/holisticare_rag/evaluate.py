"""Evaluation on a question set (JSON Lines).

Each item has ``question``, ``expect`` (``ok``, ``no_source`` or a triage category), ``sources``
(the files that must support an ``ok`` answer) and an optional ``history``.

Metrics:

* ``category_accuracy``: the answer category equals ``expect``.
* ``retrieval_hit@k``: for ``ok`` items, an expected source is in the retrieved passages.
* ``citation_precision``: for ``ok`` items, the share of citations that point to an expected source.
* ``supported_sentences``: for ``ok`` items, the share of cited sentences whose supporting quote has
  at least half of the content tokens of the sentence (a faithfulness proxy).
* ``refusal_precision`` / ``refusal_recall``: for the ``no_source`` path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .prompts import SYSTEM
from .service import Assistant
from .text import content_tokens, sentences

_CITE = re.compile(r"\[(\d+)\]")


def load_questions(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "question" not in item or "expect" not in item:
                raise ValueError(f"{path}:{n}: each item needs `question` and `expect`")
            items.append(item)
    return items


def _supported(answer) -> tuple[int, int]:
    quotes = {c.number: set(content_tokens(c.quote)) for c in answer.citations}
    ok = total = 0
    for s in sentences(answer.text):
        nums = [int(x) for x in _CITE.findall(s)]
        if not nums:
            continue
        toks = set(content_tokens(_CITE.sub("", s)))
        if not toks:
            continue
        total += 1
        if any(len(toks & quotes.get(n, set())) / len(toks) >= 0.5 for n in nums):
            ok += 1
    return ok, total


def evaluate(assistant: Assistant, items: list[dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for it in items:
        a = assistant.ask(it["question"], it.get("history"))
        exp_src = set(it.get("sources", []))
        cited = [c.source for c in a.citations]
        sup_ok, sup_total = _supported(a)
        rows.append({
            "id": it.get("id", ""), "expect": it["expect"], "got": a.category,
            "category_ok": a.category == it["expect"],
            "hit": bool(exp_src & set(a.retrieved)) if it["expect"] == "ok" else None,
            "cited": len(cited), "cited_expected": sum(c in exp_src for c in cited),
            "supported": sup_ok, "cited_sentences": sup_total, "model": a.model, "notes": "; ".join(a.notes),
        })
    df = pd.DataFrame(rows)
    ok = df[df["expect"] == "ok"]
    refused = df["got"] == "no_source"
    should = df["expect"] == "no_source"
    summary = {
        "questions": len(df),
        "category_accuracy": float(df["category_ok"].mean()),
        "retrieval_hit@k": float(ok["hit"].astype(float).mean()) if len(ok) else float("nan"),
        "citation_precision": float(ok["cited_expected"].sum() / max(1, ok["cited"].sum())),
        "supported_sentences": float(ok["supported"].sum() / max(1, ok["cited_sentences"].sum())),
        "refusal_precision": float((refused & should).sum() / max(1, refused.sum())),
        "refusal_recall": float((refused & should).sum() / max(1, should.sum())),
        "system_prompt_unchanged": SYSTEM == assistant_system_prompt(),
    }
    return summary, df


def assistant_system_prompt() -> str:
    from . import prompts  # noqa: PLC0415  (read the live constant)

    return prompts.SYSTEM
