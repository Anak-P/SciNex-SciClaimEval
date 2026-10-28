import argparse
import json
from pathlib import Path

from scinex.metrics import claim_label_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    args = ap.parse_args()
    ra = {r["id"]: r for r in json.loads(args.a.read_text())}
    rb = {r["id"]: r for r in json.loads(args.b.read_text())}
    common = sorted(set(ra) & set(rb))
    disagree = sum(ra[i]["prediction"] != rb[i]["prediction"] for i in common)
    print(f"Disagreements: {disagree}/{len(common)} ({disagree / len(common):.3f})")
    for path, recs in ((args.a, ra), (args.b, rb)):
        o = claim_label_metrics(list(recs.values()))["overall"]
        print(f"{path.stem:16s} acc {o['accuracy']:.4f}  correct {round(o['accuracy'] * o['n_samples'])}"
              f"  pairs {o['n_correct']}/{o['n_pairs']} = {o['pair_accuracy']}"
              f"  (independence: {o['accuracy'] ** 2:.3f})")


if __name__ == "__main__":
    main()
