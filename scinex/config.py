from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

SUBTASKS = ("claim_label", "claim_evidence")

# use_context values for which the preceding-sentence context x is shown.
DEFAULT_CONTEXT_VALUES = ["yes", "full paper", "true", "1"]


@dataclass
class GenerationConfig:
    """Decoding settings of the reasoning model f."""
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.05
    # Extra variables passed to the chat template. The released Qwen3-VL
    # templates ignore both keys; they are kept so the
    # template call is identical to the original runs.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Self-consistency samples N. Every config uses N=1 (greedy); for
    # N>1, sample 0 is greedy and samples 1..N-1 use temperature/top_p.
    n_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class CaptionerConfig:
    """Stage 1 of the two-stage system: the claim-blind captioner."""
    model: str = "internlm/CapRL-Qwen3VL-4B"
    max_new_tokens: int = 4096
    chat_template_kwargs: dict = field(default_factory=dict)
    # Evidence modalities that are de-rendered by Stage 1. Subtask 1 captions
    # figures only; tables bypass Stage 1.
    evidence_types: list = field(default_factory=lambda: ["figure", "table"])
    device: Optional[str] = None


@dataclass
class AdjudicationConfig:
    """Re-examination of mid-confidence answers. Only active when N>1."""
    enabled: bool = False
    contrastive: bool = True
    trace_chars: int = 1100


@dataclass
class RunConfig:
    name: str
    subtask: str
    prompts: str                                   # directory under prompts/
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # reasoning model f
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    batch_size: int = 2

    # Image preprocessing (images.preprocess)
    image_max_side: int = 2800
    image_border: float = 0.0          # white border fraction (two-stage ST1)
    composite_alpha: bool = True       # flatten transparency onto white

    # Inputs
    table_json: bool = False           # exact table grounding J(e)
    context_values: list = field(default_factory=lambda: list(DEFAULT_CONTEXT_VALUES))

    # Subtask 1 options
    other_sources_hint: bool = False   # extra note when use_context == "other sources"
    parser: str = "extended"           # Φ variant: "basic" (V1-V5) | "extended" (V6+)
    self_verify_refuted: bool = False  # second pass on every Refuted answer (N=1)
    paper_grounding_chars: int = 0     # >0: source-paper context for figures

    # Subtask 2 options
    operation_hint: bool = False       # uses the gold `operation` field (V8-V10)
    scoped_figure_context: bool = False

    # Self-consistency vote tie-break: "refuted" or "earliest" (see parsing.py)
    vote_tiebreak: str = "earliest"
    adjudication: AdjudicationConfig = field(default_factory=AdjudicationConfig)

    # Two-stage system; None = one-stage
    captioner: Optional[CaptionerConfig] = None

    # How unparseable answers enter the submission file:
    # a label to substitute, or "omit" to drop the row.
    submission_unparsed: str = "omit"

    def __post_init__(self):
        if self.subtask not in SUBTASKS:
            raise ValueError(f"subtask must be one of {SUBTASKS}, got {self.subtask!r}")
        if self.parser not in ("basic", "extended"):
            raise ValueError(f"unknown parser {self.parser!r}")
        if self.vote_tiebreak not in ("refuted", "earliest"):
            raise ValueError(f"unknown vote_tiebreak {self.vote_tiebreak!r}")
        # YAML 1.1 reads an unquoted `yes` as True; insist on quoted strings.
        if not all(isinstance(v, str) for v in self.context_values):
            raise ValueError(f"context_values must be quoted strings: {self.context_values!r}")
        self.context_values = [v.lower() for v in self.context_values]


def _build(cls, data: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        sub ={"generation": GenerationConfig, "captioner": CaptionerConfig,
               "adjudication": AdjudicationConfig}.get(key) if cls is RunConfig else None
        kwargs[key] = _build(sub, value) if sub and isinstance(value, dict) else value
    return cls(**kwargs)


def load_config(path: Path | str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _build(RunConfig, data)


def to_dict(obj) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
