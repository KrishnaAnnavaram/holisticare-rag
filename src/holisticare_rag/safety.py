"""Safety triage before retrieval, and a dose filter after generation.

The triage is a set of fixed rules. It runs on every question before any retrieval or model
call. A ``stop`` category gives a fixed reply and nothing else. A ``caution`` category adds a
fixed warning to the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_TEXT = (
    "This can be a medical emergency. Call your local emergency number now (for example 911 in the US, "
    "112 in the EU, 108 in India) or go to the nearest emergency department. Do not wait for an online answer."
)
SELF_HARM_TEXT = (
    "You do not have to deal with this alone. If you are in danger now, call your local emergency number. "
    "You can also contact a crisis line, for example 988 in the US (call or text) or the Samaritans at 116 123 "
    "in the UK and Ireland. If you can, tell someone near you how you feel."
)
DOSING_TEXT = (
    "I do not give doses or dosing schedules, for medicines or for herbal products. Ask a doctor or a "
    "pharmacist. They know your age, weight, other medicines and conditions."
)
STOP_MEDICATION_TEXT = (
    "Do not stop, change or replace a prescribed treatment without your doctor. Some medicines are dangerous "
    "to stop suddenly, and a traditional remedy is not a replacement for a prescribed treatment."
)
CAUTION_TEXT = {
    "pregnancy": "Pregnancy and breastfeeding change which treatments are safe. Ask your doctor, midwife or pharmacist first.",
    "child": "Children need different treatments and doses. Ask a paediatrician or a pharmacist first.",
}
DISCLAIMER = (
    "This is general information from the indexed documents. It is not medical advice, a diagnosis or a "
    "treatment plan. A qualified health professional must review any decision about your health."
)

_EMERGENCY = [
    r"chest pain", r"crushing (?:chest|pain)", r"can'?t breathe", r"cannot breathe", r"short(?:ness)? of breath",
    r"struggling to breathe", r"face (?:is )?droop", r"slurred speech", r"stroke", r"heart attack",
    r"unconscious", r"not breathing", r"severe bleeding", r"bleeding (?:heavily|a lot|won'?t stop)",
    r"anaphyla", r"throat (?:is )?(?:closing|swelling)", r"seizure", r"overdos", r"poison", r"choking",
    r"coughing (?:up )?blood", r"vomiting blood",
]
_SELF_HARM = [
    r"suicid", r"kill (?:my ?self|myself)", r"end (?:my|it all)", r"self[- ]?harm", r"hurt (?:my ?self|myself)",
    r"want to die", r"don'?t want to live",
]
_DOSING = [
    r"\bhow (?:much|many)\b.*\b(?:take|give|dose|mg|ml|tablet|pill|capsule|drop)", r"\bdos(?:e|age|ing)\b",
    r"\b\d+\s?(?:mg|mcg|ml|g)\b", r"\bhow often (?:should|can|do) i take",
]
_STOP_MED = [
    r"\bstop (?:taking )?(?:my )?(?:insulin|medication|medicine|meds|chemo|treatment|antidepressant|pills?)",
    r"\binstead of (?:my )?(?:insulin|medication|medicine|chemo\w*|treatment|antibiotics?|surgery)",
    r"\breplace (?:my )?(?:insulin|medication|medicine|chemo\w*|treatment)",
]
_PREGNANCY = [r"pregnan", r"breastfeed", r"breast-feed", r"nursing mother"]
_CHILD = [r"\bchild", r"\bkids?\b", r"\bbaby\b", r"\binfant", r"\btoddler", r"\bnewborn", r"\bmy son\b", r"\bmy daughter\b"]


@dataclass
class Triage:
    category: str  # ok, emergency, self_harm, dosing, stop_medication
    stop: bool
    message: str = ""
    cautions: tuple[str, ...] = ()


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def triage(question: str) -> Triage:
    t = question.lower()
    if _any(_SELF_HARM, t):
        return Triage("self_harm", True, SELF_HARM_TEXT)
    if _any(_EMERGENCY, t):
        return Triage("emergency", True, EMERGENCY_TEXT)
    if _any(_STOP_MED, t):
        return Triage("stop_medication", True, STOP_MEDICATION_TEXT)
    if _any(_DOSING, t):
        return Triage("dosing", True, DOSING_TEXT)
    cautions = tuple(k for k, pats in (("pregnancy", _PREGNANCY), ("child", _CHILD)) if _any(pats, t))
    return Triage("ok", False, "", cautions)


_DOSE_IN_TEXT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|µg|ml|g|grams?|milligrams?|tablets?|capsules?|drops?|teaspoons?|tsp|tablespoons?|tbsp)\b"
    r"|\b(?:once|twice|three times|\d+ times) (?:a|per) day\b",
    re.I,
)


def remove_doses(sentences: list[str]) -> tuple[list[str], int]:
    """Remove sentences that state an amount or a schedule. Return the kept sentences and the count removed."""
    kept = [s for s in sentences if not _DOSE_IN_TEXT.search(s)]
    return kept, len(sentences) - len(kept)
