"""Eval core: run any predictor over the test split and score it.

Pure stdlib on purpose, so the MLX env (Clef-flash) can import it too.
A predictor is any callable: PIL.Image -> P(pothole) in [0, 1].
"""
import csv, json, platform, statistics, time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# How each source's photos were taken (checked by looking at samples).
CAPTURE = {"kerala": "handheld", "taroii": "handheld", "pune": "handheld", "miia": "vehicle", "rdd_india": "vehicle",
           "rdd_japan": "vehicle", "bharat": "vehicle"}


def test_rows(split="test", limit=None):
    with open(ROOT / "data/manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return rows[:limit] if limit else rows


def auc(ys, ps):
    """ROC-AUC via the rank-sum formula (ties get average rank)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks = [0.0] * len(ps)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ps[order[j + 1]] == ps[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(ys)
    neg = len(ys) - pos
    return (sum(r for r, y in zip(ranks, ys) if y) - pos * (pos + 1) / 2) / (pos * neg)


def score(ys, ps, threshold=0.5):
    pred = [int(p >= threshold) for p in ps]
    tp = sum(1 for y, p in zip(ys, pred) if y and p)
    tn = sum(1 for y, p in zip(ys, pred) if not y and not p)
    fp = sum(1 for y, p in zip(ys, pred) if not y and p)
    fn = sum(1 for y, p in zip(ys, pred) if y and not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(ys),
        "accuracy": (tp + tn) / len(ys),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "auc": auc(ys, ps) if 0 < sum(ys) < len(ys) else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def run(name, predict, notes="", device="", limit=None, warmup=3, split="test", threshold=0.5):
    rows = test_rows(split=split, limit=limit)
    for r in rows[:warmup]:
        predict(Image.open(ROOT / r["path"]).convert("RGB"))

    ys, ps, ms = [], [], []
    for i, r in enumerate(rows, 1):
        img = Image.open(ROOT / r["path"]).convert("RGB")
        t0 = time.perf_counter()
        p = float(predict(img))
        ms.append((time.perf_counter() - t0) * 1000)
        ys.append(int(r["label"] == "pothole"))
        ps.append(p)
        if i % 100 == 0:
            print(f"  {name}: {i}/{len(rows)}")

    result = {
        "name": name,
        "notes": notes,
        "device": device,
        "machine": platform.machine() + " " + platform.platform(terse=True),
        "split": split,
        "threshold": threshold,
        "overall": score(ys, ps, threshold),
        "by_source": {},
        "by_capture": {},
        "latency_ms": {
            "p50": statistics.median(ms),
            "p95": sorted(ms)[int(0.95 * (len(ms) - 1))],
            "mean": statistics.fmean(ms),
        },
        "mistakes": [
            {"path": r["path"], "label": r["label"], "p_pothole": round(p, 4)}
            for r, y, p in zip(rows, ys, ps) if int(p >= threshold) != y
        ],
        "scores": [[r["path"], y, round(p, 5)] for r, y, p in zip(rows, ys, ps)],
    }
    for key, group in (("by_source", lambda r: r["source"]), ("by_capture", lambda r: CAPTURE[r["source"]])):
        for g in sorted({group(r) for r in rows}):
            idx = [i for i, r in enumerate(rows) if group(r) == g]
            result[key][g] = score([ys[i] for i in idx], [ps[i] for i in idx], threshold)

    out = ROOT / "results" / (f"{name}.json" if split == "test" else f"{name}.{split}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    o, l = result["overall"], result["latency_ms"]
    print(f"{name}: acc {o['accuracy']:.3f}  P {o['precision']:.3f}  R {o['recall']:.3f}  "
          f"F1 {o['f1']:.3f}  AUC {o['auc'] or 0:.3f}  p50 {l['p50']:.1f} ms  -> {out.relative_to(ROOT)}")
    return result
