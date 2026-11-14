"""Pick the cutoff for vehicle cameras when the classifier and detector are combined.

Vehicle-mode score = max(classifier, detector): pothole if either model is confident.
The cutoff is chosen on the validation split to catch --recall of vehicle potholes, then
reported on test. Needs results/{v4-384,detector}{.val,}.json from scripts/eval.py.

  python scripts/combine_vehicle.py --recall 0.85
Writes models/vehicle.json, which predict.py and serve.py read.
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pothole.metrics import auc, score


def load(name, split):
    f = ROOT / "results" / (f"{name}.json" if split == "test" else f"{name}.{split}.json")
    return {p: (y, s) for p, y, s in json.loads(f.read_text())["scores"]}


ap = argparse.ArgumentParser()
ap.add_argument("--recall", type=float, default=0.85)
a = ap.parse_args()

vehicle = lambda p: any(f"/{s}__" in p for s in ("miia", "rdd_india", "rdd_japan", "bharat"))
cls = {s: load("v4-384", s) for s in ("val", "test")}
det = {s: load("detector", s) for s in ("val", "test")}

val = [(cls["val"][p][0], max(cls["val"][p][1], det["val"][p][1])) for p in cls["val"] if vehicle(p)]
pos = sorted((s for y, s in val if y), reverse=True)
cutoff = pos[min(len(pos), max(1, math.ceil(a.recall * len(pos)))) - 1]

report = {"rule": "max(classifier, detector)", "chosen_on": "val", "target_recall": a.recall, "cutoff": round(cutoff, 4), "test": {}}
for label, keep in (("india", lambda p: "/rdd_india__" in p), ("india_bharat", lambda p: "/bharat__" in p),
                    ("japan", lambda p: "/rdd_japan__" in p), ("all_vehicle", vehicle)):
    for name, f in (("classifier_only", lambda c, d: c), ("combined", max)):
        paths = [p for p in cls["test"] if keep(p)]
        ys = [cls["test"][p][0] for p in paths]
        ps = [f(cls["test"][p][1], det["test"][p][1]) for p in paths]
        t = cutoff if name == "combined" else json.loads((ROOT / "models/v4-mnv3-384.thresholds.json").read_text())["cutoffs"]["vehicle"]
        s = score(ys, ps, t)
        c = s["confusion"]
        report["test"][f"{label}_{name}"] = {"auc": round(auc(ys, ps), 3), "caught": round(s["recall"], 3), "alerts_real": round(s["precision"], 3),
                                             "false_alarms_per_100_normal": round(100 * c["fp"] / (c["fp"] + c["tn"]), 1)}

(ROOT / "models/vehicle.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
