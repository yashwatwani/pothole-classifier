"""Is there a pothole? Runs fully offline on CPU.

  python predict.py photo.jpg
  python predict.py folder/ --json
"""
import argparse, json, sys, time
from pathlib import Path

from PIL import Image, ImageOps

from pothole.predictors import load_ours

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", action="store_true", help="one JSON object per line")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    if not files:
        sys.exit("no images found")

    predict, _, _ = load_ours()
    for f in files:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        t0 = time.perf_counter()
        p = predict(img)
        ms = (time.perf_counter() - t0) * 1000
        if a.json:
            print(json.dumps({"file": str(f), "pothole": p >= a.threshold, "probability": round(p, 4), "ms": round(ms, 1)}))
        else:
            print(f"{'POTHOLE ' if p >= a.threshold else 'no pothole'}  {p:6.1%}  {ms:5.1f} ms  {f}")


if __name__ == "__main__":
    main()
