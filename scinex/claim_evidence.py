from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import data
from .config import RunConfig
from .engine import VisionLM, free_cuda_cache
from .images import load_image
from .parsing import EVIDENCE_LABELS, OTHER, aggregate_votes, parse_evidence_label, strip_think
from .prompts import PromptSet
from .claim_label import contrast_block

log = logging.getLogger(__name__)

# Gold `operation` field → figure hint file (only used when cfg.operation_hint).
_OPERATION_HINTS = {
    "legend swap": "op_legend_swap",
    "category swap": "op_category_swap",
    "graph swap": "op_graph_swap",
    "graph flip": "op_graph_flip",
}


# ── Inputs ────────────────────────────────────────────────────────────────────

def item_id(item: dict, idx: int) -> str:
    return str(item.get("sample_id", item.get("claim_id", idx)))


def prepare_sample(item: dict, idx: int, cfg: RunConfig, data_root: Path,
                   tables_json_dir: Optional[Path]) -> dict:
    evi1, evi2 = item.get("evidence_id_1", ""), item.get("evidence_id_2", "")
    evi_type = item.get("evi_type", "unknown")
    context_use = item.get("use_context", "No")
    sample = {
        "sample_id": item_id(item, idx),
        "label": item.get("label"),            # absent on the test split
        "evi_type": evi_type,
        "context_use": context_use,
        "evidence_id_1": evi1,
        "evidence_id_2": evi2,
        "domain": item.get("domain", ""),
        "operation": item.get("operation", ""),
        "claim": item.get("claim", ""),
        "caption": item.get("caption", ""),
        "context": item.get("context", ""),
        "question": item.get("question", ""),
        "use_context": context_use.strip().lower() in cfg.context_values,
        "table1": None,
        "table2": None,
        "image1": None,
        "image2": None,
        "error": None,
    }
    try:
        sample["image1"] = load_image(data_root / evi1, cfg.image_max_side,
                                      cfg.image_border, cfg.composite_alpha)
        sample["image2"] = load_image(data_root / evi2, cfg.image_max_side,
                                      cfg.image_border, cfg.composite_alpha)
    except Exception as exc:
        sample["error"] = str(exc)
        return sample
    if evi_type.lower() == "table" and tables_json_dir:
        sample["table1"] = data.claim_evidence_table_json(tables_json_dir, evi1)
        sample["table2"] = data.claim_evidence_table_json(tables_json_dir, evi2)
    return sample


# ── Prompt construction (P2) ─────────────────────────────────────────────────

def _evidence_hint(prompts: PromptSet, cfg: RunConfig, s: dict) -> str:
    t = s["evi_type"].lower().strip()
    if t == "figure":
        key = _OPERATION_HINTS.get(s["operation"].lower().strip()) if cfg.operation_hint else None
        if key:
            return prompts["figure_hint"] + "\n\n" + prompts[key]
        return prompts["figure_hint"]
    if t == "table":
        return prompts["table_hint"]
    return ""


def _claim_lines(prompts: PromptSet, cfg: RunConfig, s: dict) -> list[str]:
    lines = [f"Claim: {s['claim']}", f"Caption: {s['caption']}"]
    if s["use_context"] and s["context"].strip():
        if cfg.scoped_figure_context and s["evi_type"].lower() == "figure":
            lines.append(f"{prompts['figure_context_label']}: {s['context']}")
        else:
            lines.append(f"Context (preceding sentences for disambiguation): {s['context']}")
    if s["question"].strip():
        lines.append(f"\nQuestion: {s['question']}")
    return lines


def _table_lines(s: dict, against: str) -> list[str]:
    if not (s["table1"] or s["table2"]):
        return []
    lines = ["\n[Structured table data for verification]"]
    for table, label in ((s["table1"], "evidence_id_1"), (s["table2"], "evidence_id_2")):
        if table:
            text = data.format_table_markdown(table, label)
            if text:
                lines.append(text)
    lines.append("Use the structured table data above to read exact cell values; "
                 "the image you identify as authentic must have values consistent "
                 f"with both {against} and the structured data.")
    return lines


def _with_images(s: dict, text: str) -> list[dict]:
    return [
        {"type": "text", "text": "evidence_id_1:\n"},
        {"type": "image", "image": s["image1"]},
        {"type": "text", "text": "\nevidence_id_2:\n"},
        {"type": "image", "image": s["image2"]},
        {"type": "text", "text": "\n\n" + text},
    ]


def one_stage_messages(prompts: PromptSet, cfg: RunConfig, s: dict) -> list[dict]:
    lines = []
    hint = _evidence_hint(prompts, cfg, s)
    if hint:
        lines += [hint, ""]
    lines += _claim_lines(prompts, cfg, s)
    lines += _table_lines(s, "the image")
    lines.append("\nVerify the claim against the two evidence images above, "
                 "following the mandatory reasoning protocol to identify the supporting evidence.")
    return [{"role": "system", "content": prompts["system"]},
            {"role": "user", "content": _with_images(s, "\n".join(lines))}]


def stage1_messages(prompts: PromptSet, s: dict, which: int) -> list[dict]:
    """Stage 1: de-render candidate ``which`` alone, without the claim."""
    image, table, label = ((s["image1"], s["table1"], "evidence_id_1") if which == 1
                           else (s["image2"], s["table2"], "evidence_id_2"))
    lines = []
    if s["caption"] and s["caption"].strip():
        lines += [f"Caption (printed with the evidence): {s['caption']}", ""]
    if table:
        text = data.format_table_markdown(table, label)
        if text:
            lines += ["[STRUCTURED TABLE DATA — authoritative source for exact cell values]",
                      text, ""]
    lines.append("Transcribe this evidence image following the headings in your "
                 "instructions. Report values, legend mappings, and labels exactly; "
                 "do not judge any claim and do not compare to any other image.")
    return [{"role": "system", "content": prompts["stage1_system"]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": "\n".join(lines)}]}]


def stage2_messages(prompts: PromptSet, cfg: RunConfig, s: dict, cap1: str, cap2: str) -> list[dict]:
    lines = []
    hint = _evidence_hint(prompts, cfg, s)
    if hint:
        lines += [hint, ""]
    lines.append("[EVIDENCE EXTRACTION 1 — claim-blind reading of evidence_id_1]")
    lines.append(cap1.strip() if cap1.strip() else "(extraction unavailable)")
    lines.append("")
    lines.append("[EVIDENCE EXTRACTION 2 — claim-blind reading of evidence_id_2]")
    lines.append(cap2.strip() if cap2.strip() else "(extraction unavailable)")
    lines.append("")
    lines += _claim_lines(prompts, cfg, s)
    lines += _table_lines(s, "its extraction")
    # Both images are always retained in Stage 2.
    lines.append("\nThe two evidence images are also attached (evidence_id_1 then "
                 "evidence_id_2). Verify each extraction against its image, then judge "
                 "the claim following the mandatory reasoning protocol.")
    return [{"role": "system", "content": prompts["stage2_system"]},
            {"role": "user", "content": _with_images(s, "\n".join(lines))}]


# ── Inference ─────────────────────────────────────────────────────────────────

class ClaimEvidenceSystem:
    def __init__(self, cfg: RunConfig, prompts: PromptSet, reasoner: VisionLM,
                 captioner: Optional[VisionLM] = None):
        self.cfg = cfg
        self.prompts = prompts
        self.reasoner = reasoner
        self.captioner = captioner

    def _generate(self, inputs, sample: bool = False) -> list[str]:
        g = self.cfg.generation
        return self.reasoner.generate(inputs, max_new_tokens=g.max_new_tokens,
                                      repetition_penalty=g.repetition_penalty, sample=sample,
                                      temperature=g.temperature, top_p=g.top_p)

    def predict_batch(self, samples: list[dict]) -> list[dict]:
        if self.captioner is None:
            return self._reason([one_stage_messages(self.prompts, self.cfg, s) for s in samples])

        # Stage 1: every candidate image in its own sequence, one batched call
        # laid out as [e1 of each sample …, e2 of each sample …].
        n = len(samples)
        convs = ([stage1_messages(self.prompts, s, 1) for s in samples]
                 + [stage1_messages(self.prompts, s, 2) for s in samples])
        try:
            inputs = self.captioner.encode(convs)
            raws = self.captioner.generate(
                inputs, max_new_tokens=self.cfg.captioner.max_new_tokens,
                repetition_penalty=self.cfg.generation.repetition_penalty)
            del inputs
            free_cuda_cache()
            caps = [strip_think(r).strip() for r in raws]
        except Exception as exc:
            log.error(f"Stage-1 captioning failed: {exc}", exc_info=True)
            caps = ["" for _ in range(2 * n)]
        cap1, cap2 = caps[:n], caps[n:]
        results = self._reason([stage2_messages(self.prompts, self.cfg, s, a, b)
                                for s, a, b in zip(samples, cap1, cap2)])
        for r, a, b in zip(results, cap1, cap2):
            r["extraction_1"], r["extraction_2"] = a, b
        return results

    def _reason(self, conversations: list[list[dict]]) -> list[dict]:
        n = self.cfg.generation.n_samples
        inputs = self.reasoner.encode(conversations)
        if n == 1:
            raws, labels = self._generate_with_retry(conversations, inputs, sample=False)
            results = [{"prediction": lbl, "confidence": 1.0 if lbl != OTHER else 0.0,
                        "all_outputs": [raw], "all_votes": [lbl]}
                       for raw, lbl in zip(raws, labels)]
        else:
            outs = [[] for _ in conversations]
            votes = [[] for _ in conversations]
            for i in range(n):
                raws, labels = self._generate_with_retry(conversations, inputs, sample=i > 0)
                for j, (raw, lbl) in enumerate(zip(raws, labels)):
                    outs[j].append(raw)
                    votes[j].append(lbl)
            results = []
            for j in range(len(conversations)):
                lbl, conf = aggregate_votes(votes[j], self.cfg.vote_tiebreak)
                results.append({"prediction": lbl, "confidence": conf,
                                "all_outputs": outs[j], "all_votes": votes[j]})
            if self.cfg.adjudication.enabled:
                self._adjudicate(conversations, results, outs, votes)
        del inputs
        free_cuda_cache()
        return results

    def _generate_with_retry(self, conversations, inputs, sample: bool):
        raws = self._generate(inputs, sample=sample)
        labels = [parse_evidence_label(r) for r in raws]
        failed = [i for i, lbl in enumerate(labels) if lbl == OTHER]
        if failed:
            retry_inputs = self.reasoner.encode([
                conversations[i] + [{"role": "assistant", "content": raws[i]},
                                    {"role": "user", "content": self.prompts["retry"]}]
                for i in failed
            ])
            retry_raws = self._generate(retry_inputs)
            del retry_inputs
            free_cuda_cache()
            for i, rraw in zip(failed, retry_raws):
                rlbl = parse_evidence_label(rraw)
                if rlbl != OTHER:
                    raws[i] = raws[i] + "\n\n[RETRY]\n" + rraw
                    labels[i] = rlbl
        return raws, labels

    def _adjudicate(self, conversations, results, outs, votes) -> None:
        adj = self.cfg.adjudication
        mid = [j for j, r in enumerate(results)
               if r["prediction"] in EVIDENCE_LABELS and r["confidence"] < 1.0]
        if not mid:
            return
        convs = []
        for j in mid:
            block = (contrast_block(outs[j], votes[j], EVIDENCE_LABELS, adj.trace_chars, "step 4")
                     if adj.contrastive else None)
            if block is not None:
                convs.append(conversations[j] + [{"role": "user", "content":
                             self.prompts["adjudicate_contrast"].format(contrast=block)}])
            else:
                convs.append(conversations[j] + [
                    {"role": "assistant", "content": outs[j][0] if outs[j] else ""},
                    {"role": "user", "content": self.prompts["adjudicate"]}])
        try:
            inputs = self.reasoner.encode(convs)
            raws = self._generate(inputs)
            del inputs
            free_cuda_cache()
        except Exception as exc:
            log.warning(f"Adjudication failed; keeping majority votes: {exc}")
            return
        for j, araw in zip(mid, raws):
            albl = parse_evidence_label(araw)
            if albl in EVIDENCE_LABELS:
                results[j]["all_outputs"] = outs[j] + ["[ADJUDICATE]\n" + araw]
                results[j]["all_votes"] = votes[j] + [albl]
                results[j]["prediction"] = albl
                _, results[j]["confidence"] = aggregate_votes(results[j]["all_votes"],
                                                              self.cfg.vote_tiebreak)


# ── Outputs ───────────────────────────────────────────────────────────────────

ID_KEY = "sample_id"


def make_record(s: dict, result: Optional[dict], n_samples: int) -> dict:
    """``result=None`` marks a sample whose images could not be loaded."""
    failed = result is None
    result = result or {"prediction": OTHER, "confidence": 0.0, "all_outputs": [], "all_votes": []}
    outs = result.get("all_outputs", [])
    rec = {
        "sample_id": s["sample_id"],
        "label": s["label"],
        "prediction": result["prediction"],
        "evidence_id_1": s["evidence_id_1"],
        "evidence_id_2": s["evidence_id_2"],
        "confidence": result["confidence"],
        "all_votes": result.get("all_votes", []),
        "evi_type": s["evi_type"],
        "context_use": s["context_use"],
        "domain": s["domain"],
        "operation": s["operation"],
        "raw_output": "" if failed else outs if n_samples > 1 else (outs[0] if outs else ""),
    }
    if "extraction_1" in result:
        rec["extraction_1"] = result["extraction_1"]
        rec["extraction_2"] = result["extraction_2"]
    return rec


def submission_rows(records: list[dict], unparsed: str) -> tuple[list[dict], int]:
    """Official task-2 format ``[{sample_id, pred_label}]``."""
    rows, fixed = [], 0
    for r in records:
        pred = r.get("prediction")
        if pred not in EVIDENCE_LABELS:
            fixed += 1
            if unparsed == "omit":
                continue
            pred = unparsed
        rows.append({"sample_id": r["sample_id"], "pred_label": pred})
    return rows, fixed
