"""Run our model over the test split and the Pune field photos, and build a browsable gallery of its calls.

  python scripts/gallery.py
Writes results/predictions_test.csv (every photo, label, probability) and
results/gallery.html (self-contained page with thumbnails; not in git, ~11 MB).
Uses what predict.py runs (handheld: classifier; vehicle: classifier + detector, with boxes)
and shows what the original v1 model said (cutoff 0.5). On RDD2022 and BharatPotHole photos it also draws the
annotators' pothole boxes, so model boxes can be checked against them. Clef-flash's verdict comes from results/clef-flash-zeroshot.json when
that run covers the current test split.
"""
import base64, csv, io, json, sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pothole import metrics
from pothole.metrics import CAPTURE
from pothole.predictors import LOADERS, load_camera_models
sys.path.insert(0, str(ROOT / "scripts"))
from train_detector import box_rows, load_photo, pothole_boxes

THUMB = 128  # ~4,400 photos must fit in a 16 MB page


def thumb(img):
    t = img.copy()
    t.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=50)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def labelled_boxes(row, labels):
    """Annotators' pothole boxes for an RDD2022 India or Japan or BharatPotHole photo, as fractions of the frame."""
    found = labels.get(Path(row["path"]).stem.split("__", 1)[-1])
    if row["source"] not in ("rdd_india", "rdd_japan", "bharat") or not found:
        return []
    img_path, label_path = found
    w, h = load_photo(img_path).size
    return [[round(float(v), 4) for v in (b[0] / w, b[1] / h, b[2] / w, b[3] / h)] for b in pothole_boxes(label_path, w, h)]


def main():
    analyze = load_camera_models()
    predict_v1, _, _ = LOADERS["v1"]()
    cut = {"handheld": json.loads((ROOT / "models/v4-mnv3-384.thresholds.json").read_text())["cutoffs"]["handheld"],
           "vehicle": json.loads((ROOT / "models/vehicle.json").read_text())["cutoff"]}
    clef_file = ROOT / "results/clef-flash-zeroshot.json"
    clef = {}
    if clef_file.exists():
        clef = {path: p >= 0.5 for path, _, p in json.loads(clef_file.read_text()).get("scores", [])}

    labels = {img.stem: (img, lab) for img, lab in box_rows("test")}
    items = []
    for r in metrics.test_rows() + metrics.test_rows("field"):
        img = Image.open(ROOT / r["path"]).convert("RGB")
        camera = CAPTURE[r["source"]]
        res = analyze(img, camera)
        w, h = img.size
        items.append({
            "file": Path(r["path"]).name, "source": r["source"], "camera": camera,
            "truth": r["label"] == "pothole", "p": res["probability"], "cut": cut[camera], "pred": res["pothole"],
            "old": predict_v1(img) >= 0.5, "clef": clef.get(r["path"]), "img": thumb(img),
            "boxes": [[round(v, 4) for v in (b[0] / w, b[1] / h, b[2] / w, b[3] / h)] for b in (x["box"] for x in res["boxes"])],
            "gt": labelled_boxes(r, labels),
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
