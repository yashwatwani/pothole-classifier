"""Pick the pothole cutoff per camera type so we catch at least --recall of potholes.

Chosen on the validation split (never the test split), then reported on test:
  python scripts/eval.py v2 --split val
  python scripts/eval.py v2
  python scripts/tune_threshold.py v2 --recall 0.95
Writes results/<name>.thresholds.json, which predict.py and serve.py read.
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pothole.metrics import CAPTURE, ROOT, score


def by_capture(scores):
    groups = {}
    for path, y, p in scores:
        source = Path(path).name.split("__")[0]
        groups.setdefault(CAPTURE[source], ([], []))
        groups[CAPTURE[source]][0].append(y)
        groups[CAPTURE[source]][1].append(p)
    return groups


def cutoff_for_recall(ys, ps, target):
    """Highest cutoff that still catches `target` of the potholes."""
    pos = sorted((p for y, p in zip(ys, ps) if y), reverse=True)
    return pos[min(len(pos), max(1, math.ceil(target * len(pos)))) - 1]


ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--recall", type=float, default=0.95, help="target share of potholes caught")
ap.add_argument("--recall-handheld", type=float, help="override --recall for handheld photos")
ap.add_argument("--recall-vehicle", type=float, help="override --recall for vehicle cameras")
a = ap.parse_args()
targets = {"handheld": a.recall_handheld or a.recall, "vehicle": a.recall_vehicle or a.recall}

val = by_capture(json.loads((ROOT / f"results/{a.name}.val.json").read_text())["scores"])
test = by_capture(json.loads((ROOT / f"results/{a.name}.json").read_text())["scores"])

cutoffs = {}
print(f"{'camera':9s} {'cutoff':>7s} | {'test: caught':>13s} {'right when alerting':>20s} {'false alarms /100 normal':>25s}")
for cap in sorted(val):
    t = cutoff_for_recall(*val[cap], targets[cap])
    cutoffs[cap] = round(t, 4)
    for label, (ys, ps) in (("@0.5", test[cap]), (f"@{t:.3f}", test[cap])):
        s = score(ys, ps, 0.5 if label == "@0.5" else t)
        c = s["confusion"]
        fa = 100 * c["fp"] / max(1, c["fp"] + c["tn"])
        print(f"{cap:9s} {label:>7s} | {s['recall']:13.1%} {s['precision']:20.1%} {fa:25.1f}")

out = ROOT / f"results/{a.name}.thresholds.json"
out.write_text(json.dumps({"target_recall": targets, "chosen_on": "val", "cutoffs": cutoffs}, indent=2))
print("->", out.relative_to(ROOT))
