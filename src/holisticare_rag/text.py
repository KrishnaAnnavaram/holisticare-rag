"""Shared text helpers: tokens, content tokens and sentences."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but
by can could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only
or other our ours out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves tell please know want get help also may might much many
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Query words that mean the same as words in the documents. Applied to queries only.
QUERY_SYNONYMS = {
    "treat": ("care", "help"), "treated": ("care", "help"), "treatment": ("care", "help"), "cure": ("care", "help"),
    "remedy": ("care", "traditional"), "remedie": ("care", "traditional"), "symptom": ("sign",), "sign": ("symptom",),
    "kid": ("children",), "child": ("children",), "doctor": ("medical",), "herb": ("herbal", "traditional"),
}


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens without stopwords and one-letter tokens, with a light plural rule."""
    out = []
    for t in tokens(text):
        if t in STOPWORDS or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def query_tokens(text: str) -> list[str]:
    """Content tokens of a query plus their synonyms (see ``QUERY_SYNONYMS``)."""
    out = content_tokens(text)
    extra = [x for t in out for x in QUERY_SYNONYMS.get(t, ())]
    return out + [x for x in extra if x not in out]


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]
