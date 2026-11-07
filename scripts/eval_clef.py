"""Zero-shot Clef-flash (Cloudflare, open weights, Jev/SystemOne-compatible) on our test split.

Runs in its own env because mlx-vlm pins its own versions:
  .venv-mlx/bin/python scripts/eval_clef.py [--limit 50]
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/clef-flash-4bit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL))
import clef_mlx
from pothole import metrics

QUESTION = {
    "pothole": {
        "type": "noul",
        "instructions": "Does this road photo show at least one pothole (a hole or broken depression in the road surface)?",
    }
}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
a = ap.parse_args()

model = clef_mlx.load(MODEL)


def predict(img):
    out = model.predict({"state": {"task": "Inspect the attached road photo."}, "images": [img], "questions": QUESTION})
    return out["pothole"]["true"]


metrics.run("clef-flash-zeroshot", predict, device="mps (mlx, 4-bit)", limit=a.limit,
            notes="Zero-shot Cloudflare clef-flash 9B (Qwen3.5-9B), 4-bit MLX, one yes/no question, no training")
