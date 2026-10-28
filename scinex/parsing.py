from __future__ import annotations

import collections
import re

OTHER = "Other"          # Φ could not extract an answer
CLAIM_LABELS = ("Supported", "Refuted")
EVIDENCE_LABELS = ("evidence_id_1", "evidence_id_2")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_LABEL = re.compile(r"###\s*Final\s+Label\s*:\s*(Supported|Refuted)", re.IGNORECASE)
_LABEL_ALT = re.compile(
    r"(?:Final\s+Label|Label|Verdict)\s*:\s*\*{0,2}\s*(Supported|Refuted)\s*\*{0,2}",
    re.IGNORECASE,
)
_LABEL_BOLD = re.compile(r"\*\*(Supported|Refuted)\*\*", re.IGNORECASE)
_EVIDENCE = re.compile(r"###\s*Final\s+Label\s*:\s*(evidence_id_1|evidence_id_2)", re.IGNORECASE)


def strip_think(raw: str) -> str:
    """Remove a ``<think>…</think>`` block."""
    return _THINK.sub("", raw).strip()


def parse_claim_label(raw: str, variant: str = "extended") -> str:
    """Φ for Subtask 1: extract Supported / Refuted.

    The last ``### Final Label:`` line wins. ``extended`` (V6 onward) then tries
    ``Label:/Verdict:`` and bold labels, and restricts the word-position
    fallback to the verdict section; ``basic`` (V1-V5) falls back directly to
    whichever label word occurs last.
    """
    text = strip_think(raw)
    matches = list(_LABEL.finditer(text))
    if matches:
        return matches[-1].group(1).capitalize()

    lower = text.lower()
    if variant == "basic":
        pos_sup, pos_ref = lower.rfind("supported"), lower.rfind("refuted")
    else:
        for pattern in (_LABEL_ALT, _LABEL_BOLD):
            alt = list(pattern.finditer(text))
            if alt:
                return alt[-1].group(1).capitalize()
        start = lower.rfind("step 4")
        if start == -1:
            start = lower.rfind("verdict")
        if start == -1:
            start = max(0, int(len(lower) * 0.7))
        verdict = lower[start:]
        pos_sup, pos_ref = verdict.rfind("supported"), verdict.rfind("refuted")
        if pos_sup == -1 and pos_ref == -1:
            pos_sup, pos_ref = lower.rfind("supported"), lower.rfind("refuted")

    if pos_sup == -1 and pos_ref == -1:
        return OTHER
    if pos_ref == -1 or (pos_sup != -1 and pos_sup > pos_ref):
        return "Supported"
    return "Refuted"


def parse_evidence_label(raw: str) -> str:
    """Φ for Subtask 2: extract evidence_id_1 / evidence_id_2.

    Falls back to whichever id is mentioned last.
    """
    text = strip_think(raw)
    matches = list(_EVIDENCE.finditer(text))
    if matches:
        return matches[-1].group(1).lower()
    lower = text.lower()
    pos_e1, pos_e2 = lower.rfind("evidence_id_1"), lower.rfind("evidence_id_2")
    if pos_e1 == -1 and pos_e2 == -1:
        return OTHER
    if pos_e2 == -1 or (pos_e1 != -1 and pos_e1 > pos_e2):
        return "evidence_id_1"
    return "evidence_id_2"


def aggregate_votes(votes: list[str], tiebreak: str = "earliest") -> tuple[str, float]:
    """Majority vote over N self-consistency samples.

    Returns the winning label and its vote share (used as confidence).
    Tie-break ``refuted`` prefers Refuted (Subtask 1, V1-V6); ``earliest``
    takes the tied label voted first, i.e. the greedy sample.
    """
    counts = collections.Counter(v for v in votes if v != OTHER)
    if not counts:
        return OTHER, 0.0
    if tiebreak == "refuted":
        winner, top = max(counts.items(), key=lambda kv: (kv[1], kv[0] == "Refuted"))
    else:
        top = max(counts.values())
        winner = next(v for v in votes if counts.get(v) == top)
    return winner, round(top / len(votes), 4)
