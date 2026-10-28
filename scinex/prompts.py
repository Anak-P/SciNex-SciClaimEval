from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptSet:
    """The prompt files of one configuration, e.g. ``PromptSet("st1/v6")``."""

    def __init__(self, name: str, root: Path = PROMPTS_DIR):
        self.name = name
        self.dir = root / name
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Prompt directory not found: {self.dir}")
        self._cache: dict[str, str] = {}

    def __contains__(self, key: str) -> bool:
        return (self.dir / f"{key}.txt").is_file()

    def __getitem__(self, key: str) -> str:
        if key not in self._cache:
            path = self.dir / f"{key}.txt"
            if not path.is_file():
                raise KeyError(f"Prompt '{key}' missing in {self.dir}")
            # read_bytes: no newline translation, so the text is exactly the file.
            self._cache[key] = path.read_bytes().decode("utf-8")
        return self._cache[key]
