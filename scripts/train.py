"""Fine-tune a small ImageNet-pretrained model (timm) to say pothole / normal.

  python scripts/train.py --model mobilenetv3_large_100 --epochs 10
Saves models/<model>.pt (best val F1) and models/<model>.onnx.
"""
import argparse, csv, json, time
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["normal", "pothole"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Images(Dataset):
    def __init__(self, split, tf):
        with open(ROOT / "data/manifest.csv") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(ROOT / r["path"]).convert("RGB")), CLASSES.index(r["label"])


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    acc = sum(int(a == b) for a, b in zip(ys, ps)) / len(ys)
    return acc, f1_score(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3_large_100")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    train_tf = T.Compose([
        T.RandomResizedCrop(a.size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((a.size, a.size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    train = DataLoader(Images("train", train_tf), batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    val = DataLoader(Images("val", eval_tf), batch_size=64, num_workers=4, persistent_workers=True)

    model = timm.create_model(a.model, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    best, history = -1, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        acc, f1 = evaluate(model, val, device)
        history.append({"epoch": ep, "loss": total / len(train.dataset), "val_acc": acc, "val_f1": f1})
        print(f"epoch {ep:2d}  loss {total / len(train.dataset):.3f}  val acc {acc:.3f}  f1 {f1:.3f}  ({time.time() - t0:.0f}s)")
        if f1 > best:
            best = f1
            torch.save({"model": a.model, "size": a.size, "classes": CLASSES, "state": model.state_dict()}, out / f"{a.model}.pt")

    ckpt = torch.load(out / f"{a.model}.pt")
    model.load_state_dict(ckpt["state"])
    model.cpu().eval()
    torch.onnx.export(model, torch.randn(1, 3, a.size, a.size), out / f"{a.model}.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    (out / f"{a.model}.train.json").write_text(json.dumps({"best_val_f1": best, "history": history}, indent=2))
    print(f"best val f1 {best:.3f} -> models/{a.model}.pt / .onnx")


if __name__ == "__main__":
    main()
