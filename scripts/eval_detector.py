"""Score the pothole detector on the India and Japan test photos: are the boxes in the right place?

  python scripts/eval_detector.py
Per country: box AP@0.5, plus the share of potholes found (a box with IoU >= 0.5 scoring at least
the cutoff) by pothole size, and how many boxes it draws on photos without potholes.
Writes results/detector.boxes.json. (Photo-level scores: python scripts/eval.py detector)
"""
import json, sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pothole import detector
from train_detector import average_precision, box_rows, pothole_boxes

SIZES = {"small (<0.8% of frame)": (0, 0.008), "medium (0.8-2%)": (0.008, 0.02), "large (>2%)": (0.02, 1)}

detect = detector.load("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(rows):
    photos = []  # (ground-truth boxes, [(score, box)], frame area)
    for img_path, label_path in rows:
        img = Image.open(img_path).convert("RGB")
        photos.append((pothole_boxes(label_path, *img.size), [(s, torch.tensor(b)) for s, b in detect(img)], img.width * img.height))

    gts = {k: g for k, (g, _, _) in enumerate(photos)}
    dets = [(s, k, b) for k, (_, d, _) in enumerate(photos) for s, b in d]
    report = {"photos": len(photos), "potholes": sum(len(g) for g in gts.values()), "box_ap50": average_precision(dets, gts)}
    for cut in (0.3, 0.5, 0.7):
        found = {name: [0, 0] for name in SIZES}
        boxes_on_clean = 0
        for g, d, frame in photos:
            kept = torch.stack([b for s, b in d if s >= cut]) if any(s >= cut for s, _ in d) else torch.zeros(0, 4)
            if not len(g):
                boxes_on_clean += len(kept)
                continue
            hit = (box_iou(g, kept).max(1).values >= 0.5) if len(kept) else torch.zeros(len(g), dtype=torch.bool)
            for b, h in zip(g, hit):
                share = float((b[2] - b[0]) * (b[3] - b[1])) / frame
                name = next(n for n, (lo, hi) in SIZES.items() if lo <= share < hi)
                found[name][0] += int(h)
                found[name][1] += 1
        n_clean = sum(1 for g, _, _ in photos if not len(g))
        report[f"score_cutoff_{cut}"] = {
            "found_by_size": {n: f"{f}/{t} ({f / max(t, 1):.0%})" for n, (f, t) in found.items()},
            "found_total": f"{sum(f for f, _ in found.values())}/{report['potholes']}",
            "boxes_per_100_clean_photos": round(100 * boxes_on_clean / n_clean, 1),
        }
    return report


report = {country: evaluate(box_rows("test", (source,))) for country, source in (("india", "rdd_india"), ("japan", "rdd_japan"))}
(ROOT / "results/detector.boxes.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
