from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def write_json(path: Path, data) -> None:
    """Write via temp file + rename so an interrupted run never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def load_checkpoint(path: Path, id_key: str) -> tuple[list[dict], set[str]]:
    """Existing predictions at ``path`` and the set of their ids."""
    path = Path(path)
    if not path.exists():
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("checkpoint is not a JSON list")
    except Exception as exc:
        log.warning(f"Could not read checkpoint {path} ({exc}); starting fresh.")
        return [], set()
    return records, {str(r[id_key]) for r in records if r.get(id_key) is not None}


def derived_path(output: Path, suffix: str) -> Path:
    """``preds.json`` → ``preds_<suffix>.json``."""
    output = Path(output)
    return output.with_name(f"{output.stem}_{suffix}.json")
