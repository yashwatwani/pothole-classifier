"""Score one model on the held-out test split.

  python scripts/eval.py ours
  python scripts/eval.py siglip-zeroshot hf-resnet50 yolov8n-pothole
  python scripts/eval.py all
Clef-flash runs in its own env: .venv-mlx/bin/python scripts/eval_clef.py
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pothole import metrics, predictors

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help=f"{list(predictors.LOADERS)} or 'all'")
ap.add_argument("--limit", type=int)
a = ap.parse_args()

for name in list(predictors.LOADERS) if a.models == ["all"] else a.models:
    predict, device, notes = predictors.LOADERS[name]()
    metrics.run(name, predict, notes=notes, device=device, limit=a.limit)
