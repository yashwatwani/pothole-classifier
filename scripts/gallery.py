"""Run our model over the test split and build a browsable gallery of its calls.

  python scripts/gallery.py
Writes results/predictions_test.csv (every photo, label, probability) and
results/gallery.html (self-contained page with thumbnails; not in git, ~7 MB).
Clef-flash's verdict is taken from results/clef-flash-zeroshot.json when present.
"""
import base64, csv, io, json, sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pothole import metrics
from pothole.predictors import load_ours

THRESHOLD = 0.5
THUMB = 300


def thumb(img):
    t = img.copy()
    t.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    predict, _, _ = load_ours()
    clef_file = ROOT / "results/clef-flash-zeroshot.json"
    clef_wrong = None
    if clef_file.exists():
        clef_wrong = {m["path"] for m in json.loads(clef_file.read_text())["mistakes"]}

    items = []
    for r in metrics.test_rows():
        img = Image.open(ROOT / r["path"]).convert("RGB")
        p = predict(img)
        truth = r["label"] == "pothole"
        clef = None
        if clef_wrong is not None:
            clef = (not truth) if r["path"] in clef_wrong else truth
        items.append({
            "file": Path(r["path"]).name, "source": r["source"], "truth": truth,
            "p": round(p, 4), "pred": p >= THRESHOLD, "clef": clef, "img": thumb(img),
        })

    with open(ROOT / "results/predictions_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "label", "p_pothole", "predicted", "correct", "clef_flash_predicted"])
        for i in items:
            w.writerow([i["file"], i["source"], "pothole" if i["truth"] else "normal", i["p"],
                        "pothole" if i["pred"] else "normal", i["pred"] == i["truth"],
                        "" if i["clef"] is None else "pothole" if i["clef"] else "normal"])

    template = (ROOT / "scripts/gallery_template.html").read_text()
    html = template.replace("/*DATA*/[]", json.dumps(items, separators=(",", ":")))
    (ROOT / "results/gallery.html").write_text(html)
    wrong = sum(i["pred"] != i["truth"] for i in items)
    print(f"{len(items)} photos, {wrong} wrong -> results/gallery.html ({len(html) / 1e6:.1f} MB), results/predictions_test.csv")


if __name__ == "__main__":
    main()
