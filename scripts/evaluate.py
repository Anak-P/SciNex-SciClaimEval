import argparse
import json
from pathlib import Path

from scinex import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--release", type=Path, default=None)
    args = ap.parse_args()
    records = json.loads(args.predictions.read_text())
    subtask = "claim_evidence" if records and "sample_id" in records[0] else "claim_label"
    dataset = json.loads(args.release.read_text()) if args.release else None
    print(json.dumps(metrics.compute(subtask, records, dataset), indent=2))


if __name__ == "__main__":
    main()
