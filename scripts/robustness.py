"""How the model holds up on bad photos: blur, dark, compressed, low-res, sideways.

  python scripts/robustness.py v2
Re-scores the test split with each corruption applied, per camera type, using the
cutoffs from results/<name>.thresholds.json. Writes results/<name>.robustness.json.
"""
import argparse, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pothole import predictors
from pothole.metrics import CAPTURE, ROOT, score, test_rows


def motion_blur(img):
    """Horizontal smear of ~9 px, like a shaky hand or a moving vehicle."""
    line = ImageFilter.Kernel((5, 5), [1 / 5 if i // 5 == 2 else 0 for i in range(25)])
    return img.filter(line).filter(line)


def dark(img):
    a = np.asarray(img, dtype=np.float32) / 255
    a = (a ** 2.2) * 0.6 + np.random.default_rng(0).normal(0, 0.03, a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def jpeg(img, q=15):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def low_res(img):
    w, h = img.size
    return img.resize((max(1, w // 4), max(1, h // 4))).resize((w, h))


CORRUPTIONS = {
    "clean": lambda im: im,
    "motion blur": motion_blur,
    "low light + noise": dark,
    "heavy JPEG (q=15)": jpeg,
    "low resolution (1/4)": low_res,
    "rotated 90°": lambda im: im.rotate(90, expand=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("name")
a = ap.parse_args()

predict, _, _ = predictors.LOADERS[a.name]()
cutoffs = json.loads((ROOT / f"results/{a.name}.thresholds.json").read_text())["cutoffs"]
rows = test_rows()
images = [Image.open(ROOT / r["path"]).convert("RGB") for r in rows]
ys = [int(r["label"] == "pothole") for r in rows]
caps = [CAPTURE[r["source"]] for r in rows]

report = {}
print(f"{'condition':22s} {'camera':9s} {'AUC':>6s} {'caught':>7s} {'right when alerting':>20s}")
for cname, fn in CORRUPTIONS.items():
    ps = [float(predict(fn(im))) for im in images]
    for cap in sorted(set(caps)):
        idx = [i for i, c in enumerate(caps) if c == cap]
        s = score([ys[i] for i in idx], [ps[i] for i in idx], cutoffs[cap])
        report.setdefault(cname, {})[cap] = s
        print(f"{cname:22s} {cap:9s} {s['auc']:6.3f} {s['recall']:7.1%} {s['precision']:20.1%}")

out = ROOT / f"results/{a.name}.robustness.json"
out.write_text(json.dumps({"cutoffs": cutoffs, "results": report}, indent=2))
print("->", out.relative_to(ROOT))
