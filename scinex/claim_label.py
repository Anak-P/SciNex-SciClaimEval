from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import data
from .config import RunConfig
from .engine import VisionLM, free_cuda_cache
from .images import load_image
from .parsing import CLAIM_LABELS, OTHER, aggregate_votes, parse_claim_label, strip_think
from .prompts import PromptSet

log = logging.getLogger(__name__)


# ── Inputs ────────────────────────────────────────────────────────────────────

def item_id(item: dict, idx: int) -> str:
    return str(item.get("claim_id", idx))


def prepare_sample(item: dict, idx: int, cfg: RunConfig, data_root: Path,
                   tables_json_dir: Optional[Path]) -> dict:
    """Load one release entry: image, J(e) text and (optionally) paper context."""
    claim_id = item_id(item, idx)
    evi_path = item.get("evi_path", "")
    evi_type = item.get("evi_type", "unknown")
    context_use = item.get("use_context", "No")
    sample = {
        "claim_id": claim_id,
        "claim_id_pair": item.get("claim_id_pair", None),
        "label": item.get("label", "Unknown"),
        "evi_type": evi_type,
        "domain": item.get("domain", ""),
        "context_use": context_use,
        "claim": item.get("claim", ""),
        "caption": item.get("caption", ""),
        "context": item.get("context", ""),
        "use_context": context_use.strip().lower() in cfg.context_values,
        "table_text": "",
        "paper_context": "",
        "image": None,
        "error": None,
    }
    if evi_type.lower() == "table" and tables_json_dir:
        sample["table_text"] = data.claim_label_table_text(tables_json_dir, evi_path, claim_id)
    if evi_type.lower() == "figure" and cfg.paper_grounding_chars > 0 and item.get("paper_path"):
        try:
            sample["paper_context"] = data.paper_context(
                data_root / item["paper_path"], sample["caption"], sample["claim"],
                cfg.paper_grounding_chars)
        except Exception:
            sample["paper_context"] = ""
    try:
        sample["image"] = load_image(data_root / evi_path, cfg.image_max_side,
                                     cfg.image_border, cfg.composite_alpha)
    except Exception as exc:
        sample["error"] = str(exc)
    return sample


# ── Prompt construction (P1) ─────────────────────────────────────────────────

def _evidence_hint(prompts: PromptSet, cfg: RunConfig, s: dict) -> str:
    t = s["evi_type"].lower().strip()
    hint = prompts["figure_hint"] if t == "figure" else prompts["table_hint"] if t == "table" else ""
    if cfg.other_sources_hint and s["context_use"].lower().strip() == "other sources":
        other = prompts["other_sources_hint"]
        hint = hint + "\n\n" + other if hint else other
    return hint


def _context_line(s: dict) -> list[str]:
    if s["use_context"] and s["context"].strip():
        return [f"Context (preceding sentences for disambiguation): {s['context']}"]
    return []


def one_stage_messages(prompts: PromptSet, cfg: RunConfig, s: dict) -> list[dict]:
    """Image + hint + J(e) + claim c + caption + context x."""
    lines = []
    hint = _evidence_hint(prompts, cfg, s)
    if hint:
        lines += [hint, ""]
    if s["table_text"]:
        lines += ["[STRUCTURED TABLE DATA — use as authoritative source for exact cell values]\n"
                  + s["table_text"], ""]
    lines += [f"Claim: {s['claim']}", f"Caption: {s['caption']}"]
    lines += _context_line(s)
    lines.append("\nVerify the claim against the evidence image above, "
                 "following the mandatory reasoning protocol.")
    return [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": [{"type": "image", "image": s["image"]},
                                     {"type": "text", "text": "\n".join(lines)}]},
    ]


def stage1_messages(prompts: PromptSet, s: dict) -> list[dict]:
    """Stage 1: claim-blind de-rendering; the claim is never included."""
    lines = []
    if s["caption"] and s["caption"].strip():
        lines += [f"Caption (printed with the evidence): {s['caption']}", ""]
    if s["table_text"]:
        lines += ["[STRUCTURED TABLE DATA — authoritative source for exact cell values]\n"
                  + s["table_text"], ""]
    lines.append("Transcribe this evidence image following the headings in your "
                 "instructions. Report values exactly; do not judge any claim.")
    return [
        {"role": "system", "content": prompts["stage1_system"]},
        {"role": "user", "content": [{"type": "image", "image": s["image"]},
                                     {"type": "text", "text": "\n".join(lines)}]},
    ]


def stage2_messages(prompts: PromptSet, cfg: RunConfig, s: dict, caption_text: str) -> list[dict]:
    """Stage 2: reasoning over the Stage-1 caption, the image and the claim."""
    lines = []
    hint = _evidence_hint(prompts, cfg, s)
    if hint:
        lines += [hint, ""]
    lines.append("[EVIDENCE EXTRACTION — claim-blind reading of the image]")
    lines.append(caption_text.strip() if caption_text.strip() else "(extraction unavailable)")
    lines.append("")
    if s["paper_context"] and s["paper_context"].strip():
        lines.append("[SOURCE-PAPER GROUNDING — what the paper says about this figure]")
        lines.append(s["paper_context"].strip())
        lines.append(prompts["paper_grounding_usage"])
        lines.append("")
    lines += [f"Claim: {s['claim']}", f"Caption: {s['caption']}"]
    lines += _context_line(s)
    # The image is always retained in Stage 2.
    lines.append("\nThe evidence image is also attached. Verify the extraction against "
                 "it, then judge the claim following the mandatory reasoning protocol.")
    return [
        {"role": "system", "content": prompts["stage2_system"]},
        {"role": "user", "content": [{"type": "image", "image": s["image"]},
                                     {"type": "text", "text": "\n".join(lines)}]},
    ]


# ── Inference ─────────────────────────────────────────────────────────────────

def _trace_tail(raw: str, max_chars: int, marker: str) -> str:
    """Verdict-bearing tail of a trace (from the last ``marker``), think-stripped."""
    t = strip_think(raw)
    cut = t.lower().rfind(marker)
    if cut == -1:
        cut = max(0, len(t) - max_chars)
    seg = t[cut:]
    if len(seg) > max_chars:
        seg = "…" + seg[-max_chars:]
    return seg.strip()


def contrast_block(raws: list[str], votes: list[str], labels: tuple[str, str],
                   max_chars: int, marker: str) -> Optional[str]:
    """'ANALYSIS A … ANALYSIS B' block from two disagreeing samples, or None."""
    a = next((r for r, v in zip(raws, votes) if v == labels[0]), None)
    b = next((r for r, v in zip(raws, votes) if v == labels[1]), None)
    if a is None or b is None:
        return None
    return (f"--- ANALYSIS A (concluded: {labels[0]}) ---\n" + _trace_tail(a, max_chars, marker)
            + f"\n\n--- ANALYSIS B (concluded: {labels[1]}) ---\n" + _trace_tail(b, max_chars, marker)
            + "\n")


class ClaimLabelSystem:
    def __init__(self, cfg: RunConfig, prompts: PromptSet, reasoner: VisionLM,
                 captioner: Optional[VisionLM] = None):
        self.cfg = cfg
        self.prompts = prompts
        self.reasoner = reasoner
        self.captioner = captioner

    def _parse(self, raw: str) -> str:
        return parse_claim_label(raw, self.cfg.parser)

    def _generate(self, inputs, sample: bool = False) -> list[str]:
        g = self.cfg.generation
        return self.reasoner.generate(inputs, max_new_tokens=g.max_new_tokens,
                                      repetition_penalty=g.repetition_penalty, sample=sample,
                                      temperature=g.temperature, top_p=g.top_p)

    def predict_batch(self, samples: list[dict]) -> list[dict]:
        if self.captioner is None:
            return self._reason([one_stage_messages(self.prompts, self.cfg, s) for s in samples])

        # Two-stage: only evidence types listed for the captioner go through Stage 1.
        types = {t.lower() for t in self.cfg.captioner.evidence_types}
        captioned = [i for i, s in enumerate(samples) if s["evi_type"].lower().strip() in types]
        captions = ["" for _ in samples]
        if captioned:
            try:
                inputs = self.captioner.encode(
                    [stage1_messages(self.prompts, samples[i]) for i in captioned])
                raws = self.captioner.generate(
                    inputs, max_new_tokens=self.cfg.captioner.max_new_tokens,
                    repetition_penalty=self.cfg.generation.repetition_penalty)
                del inputs
                free_cuda_cache()
                for i, raw in zip(captioned, raws):
                    captions[i] = strip_think(raw).strip()
            except Exception as exc:
                # Stage 2 then reasons with "(extraction unavailable)".
                log.error(f"Stage-1 captioning failed: {exc}", exc_info=True)

        captioned_set = set(captioned)
        conversations = [
            stage2_messages(self.prompts, self.cfg, s, captions[i]) if i in captioned_set
            else one_stage_messages(self.prompts, self.cfg, s)
            for i, s in enumerate(samples)
        ]
        results = self._reason(conversations)
        for r, cap in zip(results, captions):
            r["extraction"] = cap
        return results

    # Shared reasoning core: N-sample decoding, format retry, then either the
    # Refuted self-verification (N=1) or mid-confidence adjudication (N>1).
    def _reason(self, conversations: list[list[dict]]) -> list[dict]:
        n = self.cfg.generation.n_samples
        inputs = self.reasoner.encode(conversations)
        if n == 1:
            raws, labels = self._generate_with_retry(conversations, inputs, sample=False)
            if self.cfg.self_verify_refuted:
                self._self_verify_refuted(conversations, raws, labels)
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
        """Generate; re-ask once (greedy) for answers Φ cannot parse."""
        raws = self._generate(inputs, sample=sample)
        labels = [self._parse(r) for r in raws]
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
                rlbl = self._parse(rraw)
                if rlbl != OTHER:
                    raws[i] = raws[i] + "\n\n[RETRY]\n" + rraw
                    labels[i] = rlbl
        return raws, labels

    def _self_verify_refuted(self, conversations, raws: list[str], labels: list[str]) -> None:
        """Challenge every Refuted answer once; adopt Supported if the model recants."""
        refuted = [i for i, lbl in enumerate(labels) if lbl == "Refuted"]
        if not refuted:
            return
        try:
            inputs = self.reasoner.encode([
                conversations[i] + [{"role": "assistant", "content": raws[i]},
                                    {"role": "user", "content": self.prompts["self_verify_refuted"]}]
                for i in refuted
            ])
            verify_raws = self._generate(inputs)
            del inputs
            free_cuda_cache()
        except Exception as exc:
            log.error(f"Self-verification failed; keeping first-pass labels: {exc}", exc_info=True)
            return
        for i, vraw in zip(refuted, verify_raws):
            if self._parse(vraw) == "Supported":
                raws[i] = raws[i] + "\n\n[SELF-VERIFY]\n" + vraw
                labels[i] = "Supported"

    def _adjudicate(self, conversations, results, outs, votes) -> None:
        """Re-decide samples whose N votes disagree (confidence < 1)."""
        adj = self.cfg.adjudication
        mid = [j for j, r in enumerate(results)
               if r["prediction"] in CLAIM_LABELS and r["confidence"] < 1.0]
        if not mid:
            return
        convs = []
        for j in mid:
            block = (contrast_block(outs[j], votes[j], CLAIM_LABELS, adj.trace_chars, "step 3")
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
            log.error(f"Adjudication failed; keeping majority votes: {exc}", exc_info=True)
            return
        for j, araw in zip(mid, raws):
            albl = self._parse(araw)
            if albl in CLAIM_LABELS:
                results[j]["all_outputs"] = outs[j] + ["[ADJUDICATE]\n" + araw]
                results[j]["all_votes"] = votes[j] + [albl]
                results[j]["prediction"] = albl
                _, results[j]["confidence"] = aggregate_votes(results[j]["all_votes"],
                                                              self.cfg.vote_tiebreak)


# ── Outputs ───────────────────────────────────────────────────────────────────

ID_KEY = "id"


def make_record(s: dict, result: Optional[dict], n_samples: int) -> dict:
    """``result=None`` marks a sample whose image could not be loaded."""
    failed = result is None
    result = result or {"prediction": OTHER, "confidence": 0.0, "all_outputs": [], "all_votes": []}
    outs = result.get("all_outputs", [])
    rec = {
        "id": s["claim_id"],
        "claim_id_pair": s["claim_id_pair"],
        "label": s["label"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "all_votes": result.get("all_votes", []),
        "evi_type": s["evi_type"],
        "domain": s["domain"],
        "context_use": s["context_use"],
        "raw_output": "" if failed else outs if n_samples > 1 else (outs[0] if outs else ""),
    }
    if "extraction" in result:
        rec["extraction"] = result["extraction"]
    if s.get("paper_context") and not failed:
        rec["paper_grounding"] = s["paper_context"]
    return rec


def submission_rows(records: list[dict], unparsed: str) -> tuple[list[dict], int]:
    """Official task-1 format ``[{claim_id, pred_label}]``."""
    rows, fixed = [], 0
    for r in records:
        pred = str(r.get("prediction", "")).capitalize()
        if pred not in CLAIM_LABELS:
            fixed += 1
            if unparsed == "omit":
                continue
            pred = unparsed
        rows.append({"claim_id": r["id"], "pred_label": pred})
    return rows, fixed
