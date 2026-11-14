"""Train a pothole detector (draws a box around each pothole) on RDD2022 India and Japan.

  python scripts/train_detector.py --epochs 14
Faster R-CNN with a MobileNetV3-Large FPN backbone (torchvision, BSD licence, COCO-pretrained).
Uses the same India and Japan train/val/test photos as data/manifest.csv (after de-duplication) at
720 px, with RDD2022's D40 pothole boxes. Photos without potholes are kept as negatives.
Saves models/det-frcnn-mnv3.pth (best validation box AP) and a training log. To ship it, save a
half-precision copy as models/det-frcnn-mnv3.fp16.pth (what pothole/detector.py loads).
"""
import argparse, csv, json, random, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pothole.detector import build_model

RAW = ROOT / "data/raw"
SIZE = 720


def box_rows(split, sources=("rdd_india", "rdd_japan")):
    """(image path, label path) for the RDD2022 photos the manifest keeps in this split."""
    with open(ROOT / "data/manifest.csv") as f:
        rows = [(r["source"], Path(r["path"]).stem.split("__", 1)[1]) for r in csv.DictReader(f)
                if r["source"] in sources and r["split"] == split]
    out = []
    for source, stem in rows:
        if source == "rdd_india":  # YOLO labels, original split
            out.append((RAW / "rdd2022_india" / split / "images" / f"{stem}.jpg", RAW / "rdd2022_india" / split / "labels" / f"{stem}.txt"))
        else:  # Pascal VOC labels, all in the released train folder
            base = RAW / "rdd2022_world/Japan/train"
            out.append((base / "images" / f"{stem}.jpg", base / "annotations/xmls" / f"{stem}.xml"))
    return out


def pothole_boxes(label_path, w, h):
    """D40 (pothole) boxes in pixels, from a YOLO .txt (class 3) or a Pascal VOC .xml label."""
    boxes = []
    if label_path.suffix == ".xml":
        for o in ET.parse(label_path).getroot().iter("object"):
            if o.findtext("name") == "D40":
                b = o.find("bndbox")
                boxes.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    else:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts[:1] == ["3"]:
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


class Potholes(torch.utils.data.Dataset):
    def __init__(self, split, train=False):
        self.rows, self.train = box_rows(split), train
        self.jitter = ColorJitter(0.3, 0.3, 0.3, 0.05)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img_path, label_path = self.rows[i]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = pothole_boxes(label_path, w, h)
        if self.train:
            img = self.jitter(img)
            if random.random() < 0.5:
                img = F.hflip(img)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return F.to_tensor(img), {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def average_precision(dets, gts, iou=0.5):
    """VOC-style AP for one class. dets: [(score, img_id, box)], gts: {img_id: tensor[N,4]}."""
    n_gt = sum(len(g) for g in gts.values())
    used = {k: torch.zeros(len(g), dtype=torch.bool) for k, g in gts.items()}
    tp = []
    for score, img_id, box in sorted(dets, key=lambda d: -d[0]):
        g = gts[img_id]
        hit = False
        if len(g):
            ious = box_iou(box[None], g)[0]
            j = int(ious.argmax())
            if ious[j] >= iou and not used[img_id][j]:
                used[img_id][j] = hit = True
        tp.append(hit)
    tp = torch.tensor(tp, dtype=torch.float32)
    if not len(tp) or not n_gt:
        return 0.0
    ctp = tp.cumsum(0)
    recall = ctp / n_gt
    precision = ctp / torch.arange(1, len(tp) + 1)
    precision = torch.flip(torch.cummax(torch.flip(precision, [0]), 0).values, [0])  # monotone envelope
    recall = torch.cat([torch.tensor([0.0]), recall])
    return float(((recall[1:] - recall[:-1]) * precision).sum())


@torch.no_grad()
def evaluate(model, loader, device):
    """Box AP@0.5 and per-photo top score (for photo-level AUC)."""
    model.eval()
    dets, gts, photo = [], {}, []
    k = 0
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        for out, t in zip(outs, targets):
            gts[k] = t["boxes"]
            for s, b in zip(out["scores"].cpu(), out["boxes"].cpu()):
                dets.append((float(s), k, b))
            photo.append((int(len(t["boxes"]) > 0), float(out["scores"].max()) if len(out["scores"]) else 0.0))
            k += 1
    return average_precision(dets, gts), photo


def collate(batch):
    return tuple(zip(*batch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    a = ap.parse_args()

    from pothole.metrics import auc

    torch.manual_seed(0)
    random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    train = torch.utils.data.DataLoader(Potholes("train", train=True), batch_size=a.batch, shuffle=True,
                                        num_workers=4, collate_fn=collate, persistent_workers=True)
    val = torch.utils.data.DataLoader(Potholes("val"), batch_size=8, num_workers=4, collate_fn=collate,
                                      persistent_workers=True)
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train), pct_start=0.1)

    best, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for imgs, targets in train:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)  # near the peak learning rate a rare spike (~50x) blew up to NaN
            opt.step()
            sched.step()
            total += float(loss)
        box_ap, photo = evaluate(model, val, device)
        photo_auc = auc([y for y, _ in photo], [s for _, s in photo])
        history.append({"epoch": ep, "loss": total / len(train), "val_box_ap50": box_ap, "val_photo_auc": photo_auc})
        print(f"epoch {ep:2d}  loss {total / len(train):.3f}  val box AP@0.5 {box_ap:.3f}  photo AUC {photo_auc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if box_ap > best:
            best = box_ap
            torch.save(model.state_dict(), ROOT / "models/det-frcnn-mnv3.pth")
    (ROOT / "models/det-frcnn-mnv3.train.json").write_text(json.dumps({"best_val_box_ap50": best, "history": history}, indent=2))
    print(f"best val box AP@0.5 {best:.3f} -> models/det-frcnn-mnv3.pth")


if __name__ == "__main__":
    main()
