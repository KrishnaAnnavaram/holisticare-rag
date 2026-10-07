"""History-aware retrieval query.

A follow-up such as "what about for children?" has few content tokens. The retrieval query
then adds the content tokens of the earlier user questions (newest first), up to a limit.
The question that the model answers does not change. Only the retrieval query changes.
"""
from __future__ import annotations

import re

from .text import content_tokens

FOLLOW_UP = re.compile(r"^\s*(?:and|what about|how about|is it|does it|can it|what if|also|and for|for)\b", re.I)
PRONOUN = re.compile(r"\b(?:it|this|that|they|them|those|these)\b", re.I)


def is_follow_up(question: str) -> bool:
    return bool(FOLLOW_UP.search(question)) or (len(content_tokens(question)) <= 3 and bool(PRONOUN.search(question)))


def retrieval_query(question: str, history: list[dict], max_added: int = 8) -> str:
    """The text to retrieve with. ``history`` is a list of ``{"role": ..., "content": ...}`` messages."""
    if not history or not (is_follow_up(question) or len(content_tokens(question)) <= 2):
        return question
    own = set(content_tokens(question))
    added: list[str] = []
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        for t in content_tokens(msg.get("content", "")):
            if t not in own and t not in added:
                added.append(t)
        if len(added) >= max_added:
            break
    return question if not added else f"{question} {' '.join(added[:max_added])}"
