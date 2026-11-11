"""Run our model over the test split and build a browsable gallery of its calls.

  python scripts/gallery.py
Writes results/predictions_test.csv (every photo, label, probability) and
results/gallery.html (self-contained page with thumbnails; not in git, ~11 MB).
Uses the default model with its per-camera cutoffs, and shows what the original v1 model
said (cutoff 0.5). Clef-flash's verdict comes from results/clef-flash-zeroshot.json when
that run covers the current test split.
"""
import base64, csv, io, json, sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pothole import metrics
from pothole.metrics import CAPTURE
from pothole.predictors import LOADERS, cutoffs, load_ours

THUMB = 200  # 1,770 photos must fit in a 16 MB page


def thumb(img):
    t = img.copy()
    t.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=62)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    predict, _, _ = load_ours()
    predict_v1, _, _ = LOADERS["v1"]()
    cut = cutoffs()
    clef_file = ROOT / "results/clef-flash-zeroshot.json"
    clef = {}
    if clef_file.exists():
        clef = {path: p >= 0.5 for path, _, p in json.loads(clef_file.read_text()).get("scores", [])}

    items = []
    for r in metrics.test_rows():
        img = Image.open(ROOT / r["path"]).convert("RGB")
        p = predict(img)
        camera = CAPTURE[r["source"]]
        items.append({
            "file": Path(r["path"]).name, "source": r["source"], "camera": camera,
            "truth": r["label"] == "pothole", "p": round(p, 4), "cut": cut[camera], "pred": p >= cut[camera],
            "old": predict_v1(img) >= 0.5, "clef": clef.get(r["path"]), "img": thumb(img),
        })

    with open(ROOT / "results/predictions_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "camera", "label", "p_pothole", "cutoff", "predicted", "correct",
                    "v1_predicted", "clef_flash_predicted"])
        name = lambda b: "" if b is None else "pothole" if b else "normal"
        for i in items:
            w.writerow([i["file"], i["source"], i["camera"], name(i["truth"]), i["p"], i["cut"], name(i["pred"]),
                        i["pred"] == i["truth"], name(i["old"]), name(i["clef"])])

    template = (ROOT / "scripts/gallery_template.html").read_text()
    html = template.replace("/*DATA*/[]", json.dumps(items, separators=(",", ":")))
    (ROOT / "results/gallery.html").write_text(html)
    wrong = sum(i["pred"] != i["truth"] for i in items)
    print(f"{len(items)} photos, {wrong} wrong -> results/gallery.html ({len(html) / 1e6:.1f} MB), results/predictions_test.csv")


if __name__ == "__main__":
    main()
