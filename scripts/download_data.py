"""Download the three open datasets into data/raw/ (about 450 MB).

  python scripts/download_data.py
Then run scripts/prepare_data.py. Add --clef to also fetch the 6.2 GB Clef-flash 4-bit
weights into models/clef-flash-4bit for scripts/eval_clef.py.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATASETS = [
    "mabughali/miia-pothole-train",
    "taroii/pothole-detection",
    "Arpitraj01/Pothole_classification",
]

for repo in DATASETS:
    out = ROOT / "data/raw" / repo.replace("/", "__")
    snapshot_download(repo, repo_type="dataset", local_dir=out)
    print("ok", out.relative_to(ROOT))

if "--clef" in sys.argv:
    out = ROOT / "models/clef-flash-4bit"
    snapshot_download("mlx-community/clef-flash-4bit", local_dir=out)
    print("ok", out.relative_to(ROOT))
