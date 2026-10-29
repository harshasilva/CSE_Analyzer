"""Score a headline as positive, negative, or neutral.

This is the text scorer used by the price model. It reads the words in a
headline. It does not use the old TF-IDF classifier.
"""

from __future__ import annotations

import re
from datetime import date

from cse_common import parse_cse_date

POSITIVE = {
    "advance",
    "advanced",
    "approval",
    "approved",
    "beat",
    "beats",
    "boost",
    "boosted",
    "bullish",
    "dividend",
    "expand",
    "expansion",
    "gain",
    "gains",
    "growth",
    "higher",
    "improve",
    "improved",
    "improvement",
    "jump",
    "jumped",
    "optimistic",
    "outperform",
    "profit",
    "profits",
    "rally",
    "rallied",
    "rebound",
    "rebounded",
    "record",
    "recover",
    "recovery",
    "rise",
    "rises",
    "rising",
    "rose",
    "soar",
    "soared",
    "soars",
    "strong",
    "stronger",
    "strengthens",
    "strengthened",
    "surplus",
    "surge",
    "surged",
    "surges",
    "upbeat",
    "upgrade",
}

NEGATIVE = {
    "bankruptcy",
    "bearish",
    "concern",
    "concerns",
    "crisis",
    "decline",
    "declined",
    "declines",
    "default",
    "deficit",
    "delay",
    "delayed",
    "depreciated",
    "depreciation",
    "downgrade",
    "downgraded",
    "drop",
    "dropped",
    "drops",
    "fall",
    "falls",
    "fell",
    "fraud",
    "lawsuit",
    "loss",
    "losses",
    "miss",
    "missed",
    "misses",
    "penalty",
    "plunge",
    "plunged",
    "probe",
    "recession",
    "shrink",
    "slump",
    "slumped",
    "suspend",
    "suspended",
    "warning",
    "weak",
    "weakened",
    "weakens",
    "weaker",
    "sinks",
    "sank",
    "tumbled",
    "tumbles",
}

_WORD = re.compile(r"[a-z]+")
_MONTH_FIRST = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def score_text(text: str) -> float:
    """Return a score from -1 (negative) to 1 (positive). Neutral text is 0."""
    tokens = _WORD.findall(str(text).lower())
    if not tokens:
        return 0.0
    positive = sum(token in POSITIVE for token in tokens)
    negative = sum(token in NEGATIVE for token in tokens)
    if positive + negative == 0:
        return 0.0
    return (positive - negative) / (positive + negative)


def label_from_score(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def extract_date(value) -> date | None:
    """Read a date from a CSE field or from a headline such as 'May 11, 2026'."""
    parsed = parse_cse_date(value)
    if parsed is not None:
        return parsed
    match = _MONTH_FIRST.search(str(value or ""))
    if not match:
        return None
    month = _MONTHS[match.group(1).lower()[:3]]
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None
