from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional


def release_path(data_root: Path, split: str, subtask: str) -> Path:
    task = 1 if subtask == "claim_label" else 2
    return Path(data_root) / f"{split}_task{task}_release.json"


def load_release(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Exact table grounding J(e) ─────────────────────────────────────────────────
# Only used on the development split; official runs pass --png_only.

def claim_label_table_text(tables_json_dir: Path, evi_path: str, claim_id: str) -> str:
    """Subtask 1: J(e) for a table, rendered as ``Caption/Columns/Rows`` text.

    Returns "" when no file exists or it cannot be read (the image is then the
    only channel). The first existing candidate is used even if unreadable.
    """
    split_dir = Path(evi_path).parent.name if evi_path else ""
    candidates = list(dict.fromkeys(
        ([tables_json_dir / split_dir / f"{claim_id}.json"] if split_dir else [])
        + [tables_json_dir / "dev" / f"{claim_id}.json", tables_json_dir / f"{claim_id}.json"]
    ))
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                table = json.load(f)
            lines = []
            if table.get("table_caption"):
                lines.append(f"Caption: {table['table_caption']}")
            if table.get("table_column_names"):
                lines.append(f"Columns: {' | '.join(str(c) for c in table['table_column_names'])}")
            if table.get("table_content_values"):
                lines.append("Rows:")
                for row in table["table_content_values"]:
                    lines.append(f"  {' | '.join(str(c) for c in row)}")
            return "\n".join(lines)
        except Exception:
            return ""
    return ""


def claim_evidence_table_json(tables_json_dir: Path, evi_path: str) -> Optional[dict]:
    """Subtask 2: raw J(e) dict for one candidate image, or None."""
    if not evi_path:
        return None
    fname = os.path.basename(evi_path).replace(".png", ".json")
    split_dir = Path(evi_path).parent.name
    candidates = [tables_json_dir / split_dir / fname, tables_json_dir / fname] if split_dir \
        else [tables_json_dir / fname]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def format_table_markdown(table: dict, label: str) -> str:
    """Subtask 2: render J(e) as a Markdown table headed by the candidate id."""
    try:
        caption = table.get("table_caption", "").strip()
        cols = table.get("table_column_names", [])
        rows = table.get("table_content_values", [])
        if not cols and not rows:
            return ""
        lines = [f"[Structured table for {label}]"]
        if caption:
            lines.append(f"Caption: {caption}")
        if cols:
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)
    except Exception:
        return ""


# ── Source-paper context for figures (two-stage Subtask 1) ────────────────────

_PAPER_CACHE: dict[str, list[str]] = {}
_PAPER_LOCK = threading.Lock()
_STOPWORDS = set(
    "the a an of to and in for is are on with as by we our this that be can from at "
    "it its their than then also more most into using use used between over under "
    "after before only not no all each both us them they these those which what when "
    "where how figure fig table shows show shown see seen results result method "
    "methods model models compared comparison different values value".split()
)


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9\-]+", (s or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _paper_paragraphs(paper_path: Path) -> list[str]:
    key = str(paper_path)
    with _PAPER_LOCK:
        if key in _PAPER_CACHE:
            return _PAPER_CACHE[key]
    paras: list[str] = []
    try:
        with open(paper_path, "r", encoding="utf-8") as f:
            paper = json.load(f)
        for sec in paper.get("full_text", []):
            for par in sec.get("paragraphs", []):
                if par and par.strip():
                    paras.append(par.strip())
    except Exception:
        paras = []
    with _PAPER_LOCK:
        _PAPER_CACHE[key] = paras
    return paras


def paper_context(paper_path: Path, caption: str, claim: str, max_chars: int) -> str:
    """Paragraphs of the source paper that discuss this figure.

    Picks every paragraph citing "Figure N" (N from the caption); otherwise the
    single paragraph with the largest content-word overlap with claim+caption
    (at least 2 words). Truncated to ``max_chars`` at a word boundary.
    """
    paras = _paper_paragraphs(paper_path)
    if not paras:
        return ""
    picked: list[str] = []
    m = re.search(r"Fig(?:ure|\.)?\s*(\d+)", caption or "")
    if m:
        pat = re.compile(rf"Fig(?:ure|\.)?\s*{m.group(1)}\b")
        picked = [p for p in paras if pat.search(p)]
    if not picked:
        query = _tokens(claim) | _tokens(caption)
        if query:
            best_score, best = 0, None
            for p in paras:
                score = len(_tokens(p) & query)
                if score > best_score:
                    best_score, best = score, p
            if best is not None and best_score >= 2:
                picked = [best]
    if not picked:
        return ""
    text = "\n".join(picked)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return text


# ── Ordered, bounded CPU prefetch ─────────────────────────────────────────────

def prefetch(items: Iterable, fn: Callable, workers: int = 6, depth: int = 8) -> Iterator:
    """Yield ``fn(*item)`` in input order, loading up to ``depth`` items ahead.

    ``None`` results (already-processed samples) are skipped.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefetch") as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.submit(fn, *item))
            if len(pending) >= depth:
                out = pending.popleft().result()
                if out is not None:
                    yield out
        while pending:
            out = pending.popleft().result()
            if out is not None:
                yield out
