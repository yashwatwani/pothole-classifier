"""Build one binary dataset (pothole / normal) from the open datasets in scripts/download_data.py.

Sources:
  mabughali/miia-pothole-train      4,026 dashcam images, South Africa, Apache-2.0
  taroii/pothole-detection            437 images, already split
  Arpitraj01/Pothole_classification   ~490 images, Kerala, MIT (severity → binary)
  aiswarya9302/RDD_2022India        7,706 vehicle-camera images, India, from RDD2022
                                    (Arya et al. 2022), CC BY-SA 4.0. Pothole = any D40 box;
                                    crack-only and undamaged photos count as normal.
  RDD2022 Japan                    10,506 vehicle-camera images, same labels and licence. No split is
                                    given, and neighbouring photos come from the same drive, so blocks
                                    of 100 consecutive photos are split 70/15/15.
  QR4Change (Mendeley)                389 pothole photos taken on two phones in Pune, CC BY 4.0. Test only
                                    (split "field"): the set's no-pothole photos come from other cameras,
                                    so training on them would teach camera, not pothole.

Output: data/images/{train,val,test,field}/{pothole,normal}/*.jpg + data/manifest.csv
Images are resized to max 512 px and near-duplicates (perceptual hash) are dropped,
so the same photo can't sit in both train and test.
"""
import csv, io, random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "images"
MAX_SIDE = 512
SPLIT_ORDER = {"test": 0, "val": 1, "field": 2, "train": 3}  # dupes keep the test/val/field copy


def items():
    # MIIA: no split given -> stratified 70/15/15
    t = pq.read_table(RAW / "mabughali__miia-pothole-train/data/train-00000-of-00001.parquet").to_pylist()
    rng = random.Random(42)
    for label in (0, 1):
        rows = [r for r in t if r["label"] == label]
        rng.shuffle(rows)
        n = len(rows)
        for i, r in enumerate(rows):
            split = "test" if i < 0.15 * n else "val" if i < 0.30 * n else "train"
            yield "miia", split, "pothole" if label else "normal", r["image"]["path"], lambda r=r: r["image"]["bytes"]

    base = RAW / "taroii__pothole-detection"
    for split_dir, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        for cls_dir, cls in (("pothole", "pothole"), ("no pothole", "normal")):
            for p in sorted((base / split_dir / cls_dir).glob("*")):
                yield "taroii", split, cls, p.name, p.read_bytes

    base = RAW / "Arpitraj01__Pothole_classification/datasets"
    for split in ("train", "val", "test"):
        for sev in ("none", "low", "medium", "severe"):
            for p in sorted((base / split / sev).glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    yield "kerala", split, "normal" if sev == "none" else "pothole", p.name, p.read_bytes

    base = RAW / "rdd2022_india"
    for split in ("train", "val", "test"):
        for p in sorted((base / split / "images").glob("*.jpg")):
            boxes = (base / split / "labels" / f"{p.stem}.txt").read_text().split("\n")
            pothole = any(b.split()[:1] == ["3"] for b in boxes)
            yield "rdd_india", split, "pothole" if pothole else "normal", p.name, p.read_bytes

    base = RAW / "rdd2022_world/Japan/train"
    photos = sorted((base / "images").glob("*.jpg"))
    blocks = [photos[i:i + 100] for i in range(0, len(photos), 100)]
    random.Random(42).shuffle(blocks)
    for i, block in enumerate(blocks):
        split = "test" if i < 0.15 * len(blocks) else "val" if i < 0.30 * len(blocks) else "train"
        for p in block:
            names = [o.findtext("name") for o in ET.parse(base / "annotations/xmls" / f"{p.stem}.xml").getroot().iter("object")]
            yield "rdd_japan", split, "pothole" if "D40" in names else "normal", p.name, p.read_bytes

    for p in sorted((RAW / "qr4change_pune/pothole/yes").glob("*.jpg")):
        if str(Image.open(p).getexif().get(271, "")).startswith("realme"):  # the Pune survey phones
            yield "pune", "field", "pothole", p.name, p.read_bytes


def main():
    loaded = []
    for source, split, cls, name, get in items():
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(get()))).convert("RGB")
        except Exception as e:
            print("skip", source, name, e)
            continue
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        h = int(str(imagehash.phash(img)), 16)
        loaded.append((SPLIT_ORDER[split], source, split, cls, name, img, h))

    loaded.sort(key=lambda x: x[0])
    seen, n_seen, rows, dropped = np.zeros(len(loaded), dtype=np.uint64), 0, [], Counter()
    for _, source, split, cls, name, img, h in loaded:
        if n_seen and np.bitwise_count(seen[:n_seen] ^ np.uint64(h)).min() <= 4:
            dropped[source] += 1
            continue
        seen[n_seen] = h
        n_seen += 1
        dest = OUT / split / cls / f"{source}__{Path(name).stem}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, quality=92)
        rows.append({"path": str(dest.relative_to(ROOT)), "split": split, "label": cls, "source": source})

    with open(ROOT / "data" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "label", "source"])
        w.writeheader()
        w.writerows(rows)

    print("near-duplicates dropped:", dict(dropped))
    for split in SPLIT_ORDER:
        c = Counter((r["label"]) for r in rows if r["split"] == split)
        s = Counter((r["source"]) for r in rows if r["split"] == split)
        print(f"{split:5s} {dict(c)}  by source {dict(s)}")


if __name__ == "__main__":
    main()
