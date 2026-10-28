from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import claim_evidence, claim_label, data, metrics
from .config import load_config, to_dict
from .engine import VisionLM, load_vlm
from .outputs import derived_path, load_checkpoint, write_json
from .parsing import OTHER
from .prompts import PromptSet

log = logging.getLogger("scinex")

TASKS = {
    "claim_label": (claim_label, claim_label.ClaimLabelSystem),
    "claim_evidence": (claim_evidence, claim_evidence.ClaimEvidenceSystem),
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciNex runs for NTCIR-19 SciClaimEval")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True,
                   help="The dataset's data/ directory (contains *_release.json).")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data_path", type=Path, default=None,
                   help="Release JSON to use instead of <data_root>/<split>_taskN_release.json.")
    p.add_argument("--png_only", action="store_true",
                   help="Never inject structured table values J(e): the official "
                        "submission setting. Required for the test split.")
    p.add_argument("--model", default=None, help="Override the reasoner checkpoint (path or HF id).")
    p.add_argument("--captioner", default=None, help="Override the Stage-1 captioner checkpoint.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--captioner_device", default=None)
    p.add_argument("--n_samples", type=int, default=None,
                   help="Override self-consistency N (default: 1).")
    p.add_argument("--limit", type=int, default=None, help="Only the first N samples (smoke tests).")
    p.add_argument("--workers", type=int, default=6, help="Image-loading threads.")
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args(argv)


def setup_logging(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(output.with_suffix(".log"), encoding="utf-8")],
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.model:
        cfg.model = args.model
    if args.captioner and cfg.captioner:
        cfg.captioner.model = args.captioner
    if args.n_samples:
        cfg.generation.n_samples = args.n_samples
    if args.split == "test" and cfg.table_json and not args.png_only:
        sys.exit("The test split must be run with --png_only (task rule).")

    setup_logging(args.output)
    module, system_cls = TASKS[cfg.subtask]
    tables_json_dir = (args.data_root / "tables_json"
                       if cfg.table_json and not args.png_only else None)
    data_path = args.data_path or data.release_path(args.data_root, args.split, cfg.subtask)
    log.info(f"Config {cfg.name} ({args.config}) · split={args.split} · data={data_path}")
    log.info(f"Table channel J(e): {tables_json_dir or 'DISABLED (PNG-only)'}")

    dataset = data.load_release(data_path)
    if args.limit:
        dataset = dataset[:args.limit]
    records, done = load_checkpoint(args.output, module.ID_KEY)
    if done:
        log.info(f"Resuming: {len(done)} of {len(dataset)} samples already predicted.")

    # Subtask 1 scripts gathered images per conversation, Subtask 2 per batch.
    per_conv = cfg.subtask == "claim_label"
    reasoner = load_vlm(cfg.model, args.device,
                        chat_template_kwargs=cfg.generation.chat_template_kwargs,
                        vision_per_conversation=per_conv)
    captioner = None
    if cfg.captioner:
        cap_device = args.captioner_device or cfg.captioner.device or args.device
        if cfg.captioner.model == cfg.model and cap_device == args.device:
            # Same checkpoint for both stages: share the weights.
            captioner = VisionLM(reasoner.model, reasoner.processor,
                                 chat_template_kwargs=cfg.captioner.chat_template_kwargs,
                                 vision_per_conversation=per_conv)
        else:
            captioner = load_vlm(cfg.captioner.model, cap_device,
                                 chat_template_kwargs=cfg.captioner.chat_template_kwargs,
                                 vision_per_conversation=per_conv)
    system = system_cls(cfg, PromptSet(cfg.prompts), reasoner, captioner)

    def prepare(item, idx):
        if module.item_id(item, idx) in done:
            return None
        return module.prepare_sample(item, idx, cfg, args.data_root, tables_json_dir)

    n = cfg.generation.n_samples
    since_save, n_new = 0, 0

    def flush(batch: list[dict]) -> None:
        nonlocal since_save, n_new
        valid = [s for s in batch if not s["error"]]
        for s in batch:
            if s["error"]:
                log.error(f"Sample {module.item_id(s, -1)}: could not load evidence: {s['error']}")
                records.append(module.make_record(s, None, n))
        if valid:
            try:
                results = system.predict_batch(valid)
            except Exception as exc:
                log.error(f"Batch inference failed: {exc}", exc_info=True)
                results = [{"prediction": OTHER, "confidence": 0.0,
                            "all_outputs": [], "all_votes": []} for _ in valid]
            records.extend(module.make_record(s, r, n) for s, r in zip(valid, results))
        since_save += len(batch)
        n_new += len(batch)
        if since_save >= args.save_every:
            write_json(args.output, records)
            since_save = 0

    t0 = time.time()
    batch: list[dict] = []
    remaining = len(dataset) - len(done)
    for sample in data.prefetch(((item, i) for i, item in enumerate(dataset)), prepare,
                                workers=args.workers):
        batch.append(sample)
        if len(batch) >= cfg.batch_size:
            flush(batch)
            batch = []
            log.info(f"{n_new}/{remaining} samples")
    if batch:
        flush(batch)
    elapsed = time.time() - t0
    write_json(args.output, records)
    log.info(f"Predictions → {args.output}")

    rows, unparsed = module.submission_rows(records, cfg.submission_unparsed)
    write_json(derived_path(args.output, "submission"), rows)
    if unparsed:
        log.warning(f"{unparsed} unparseable answer(s): "
                    f"{'omitted' if cfg.submission_unparsed == 'omit' else 'set to ' + cfg.submission_unparsed}"
                    " in the submission file.")

    write_json(derived_path(args.output, "run"), {
        "config_file": str(args.config), "config": to_dict(cfg), "split": args.split,
        "png_only": tables_json_dir is None, "n_predicted_this_run": n_new,
        "seconds": round(elapsed, 1),
        "seconds_per_sample": round(elapsed / n_new, 2) if n_new else None,
    })

    scores = metrics.compute(cfg.subtask, records, dataset)
    if scores:
        write_json(derived_path(args.output, "metrics"), scores)
        log.info(f"Overall: {scores['overall']}")
    else:
        log.info("No gold labels in this split — metrics skipped.")


if __name__ == "__main__":
    main()
