from __future__ import annotations

from typing import Optional

from .parsing import CLAIM_LABELS, EVIDENCE_LABELS

_NO_PAIR = ("no pair", "none", "")


def _acc(recs: list[dict], gold: str = "label", pred: str = "prediction") -> Optional[float]:
    return round(sum(r[gold] == r[pred] for r in recs) / len(recs), 4) if recs else None


# ── Subtask 1 ─────────────────────────────────────────────────────────────────

def _cap(v) -> str:
    return str(v or "").capitalize()


def _prf(y_true, y_pred, cls):
    tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
    fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
    fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def claim_label_scores(records: list[dict]) -> dict:
    """Accuracy and macro-F1. Unparseable answers count as wrong."""
    pairs = [(_cap(r["label"]), _cap(r["prediction"])) for r in records
             if _cap(r.get("label")) in CLAIM_LABELS]
    if not pairs:
        return {}
    y_true, y_pred = zip(*pairs)
    sp, sr, sf = _prf(y_true, y_pred, "Supported")
    rp, rr, rf = _prf(y_true, y_pred, "Refuted")
    return {
        "n_samples": len(y_true),
        "accuracy": round(sum(t == p for t, p in pairs) / len(pairs), 4),
        "macro_f1": round((sf + rf) / 2, 4),
        "per_class": {
            "supported": {"p": round(sp, 4), "r": round(sr, 4), "f1": round(sf, 4)},
            "refuted": {"p": round(rp, 4), "r": round(rr, 4), "f1": round(rf, 4)},
        },
    }


def pair_accuracy(records: list[dict], pair_lookup: Optional[dict] = None) -> dict:
    """Fraction of complete pairs (≥2 members) with every member correct."""
    pair_lookup = pair_lookup or {}
    groups: dict[str, list[bool]] = {}
    for r in records:
        if _cap(r.get("label")) not in CLAIM_LABELS:
            continue
        key = r.get("claim_id_pair") or pair_lookup.get(str(r.get("id")))
        if not key or str(key).lower() in _NO_PAIR:
            continue
        groups.setdefault(key, []).append(_cap(r["prediction"]) == _cap(r["label"]))
    complete = [v for v in groups.values() if len(v) >= 2]
    if not complete:
        return {"n_pairs": 0, "n_correct": 0, "pair_accuracy": None}
    correct = sum(all(v) for v in complete)
    return {"n_pairs": len(complete), "n_correct": correct,
            "pair_accuracy": round(correct / len(complete), 4)}


def claim_label_metrics(records: list[dict], dataset: Optional[list[dict]] = None) -> dict:
    lookup = {str(d["claim_id"]): d.get("claim_id_pair") for d in (dataset or [])
              if d.get("claim_id") is not None}
    scored = [r for r in records if _cap(r.get("label")) in CLAIM_LABELS]
    if not scored:
        return {}
    out = {"overall": {**claim_label_scores(scored), **pair_accuracy(scored, lookup)}}
    cm = {g: {"Supported": 0, "Refuted": 0, "Other": 0} for g in CLAIM_LABELS}
    for r in scored:
        p = _cap(r["prediction"])
        cm[_cap(r["label"])][p if p in CLAIM_LABELS else "Other"] += 1
    out["overall"]["confusion_matrix"] = cm
    for field, values in (("evi_type", ("table", "figure")), ("domain", ("nlp", "ml", "peerj"))):
        out[f"by_{field}"] = {}
        for v in values:
            sub = [r for r in scored if str(r.get(field, "")).lower() == v]
            if sub:
                out[f"by_{field}"][v] = {**claim_label_scores(sub), **pair_accuracy(sub, lookup)}
    return out


# ── Subtask 2 ─────────────────────────────────────────────────────────────────

def claim_evidence_metrics(records: list[dict]) -> dict:
    """Accuracy over samples with a gold label and a parsed answer.

    Unparseable answers are excluded from the denominator, as in the original
    scripts; ``n_unparsed`` reports how many.
    """
    gold = [r for r in records if r.get("label") in EVIDENCE_LABELS]
    if not gold:
        return {}
    scored = [r for r in gold if r.get("prediction") in EVIDENCE_LABELS]

    def block(recs):
        errors = [r for r in recs if r["prediction"] != r["label"]]
        return {
            "n_samples": len(recs),
            "accuracy": _acc(recs),
            "errors_gold_e1": sum(r["label"] == "evidence_id_1" for r in errors),
            "errors_gold_e2": sum(r["label"] == "evidence_id_2" for r in errors),
        }

    out = {"overall": {**block(scored), "n_unparsed": len(gold) - len(scored)}}
    for field, values in (("evi_type", ("table", "figure")), ("domain", ("nlp", "ml", "peerj"))):
        out[f"by_{field}"] = {}
        for v in values:
            sub = [r for r in scored if str(r.get(field, "")).lower() == v]
            if sub:
                out[f"by_{field}"][v] = block(sub)
    return out


def compute(subtask: str, records: list[dict], dataset: Optional[list[dict]] = None) -> dict:
    if subtask == "claim_label":
        return claim_label_metrics(records, dataset)
    return claim_evidence_metrics(records)
