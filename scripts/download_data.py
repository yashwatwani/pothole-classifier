"""Download the four open datasets into data/raw/ (about 900 MB).

  python scripts/download_data.py
Then run scripts/prepare_data.py. Add --clef to also fetch the 6.2 GB Clef-flash 4-bit
weights into models/clef-flash-4bit for scripts/eval_clef.py.
"""
import sys, zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

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

# RDD2022 India (Arya et al. 2022, CC BY-SA 4.0): YOLO labels, class 3 = pothole.
zip_path = hf_hub_download("aiswarya9302/RDD_2022India", "RDD2022_India.zip", repo_type="dataset",
                           local_dir=ROOT / "data/raw/rdd2022_india_zip")
out = ROOT / "data/raw/rdd2022_india"
with zipfile.ZipFile(zip_path) as z:
    members = [m for m in z.namelist() if m.split("/")[0] in ("train", "val", "test") and ".." not in m]
    z.extractall(out, members)
print("ok", out.relative_to(ROOT), f"({len(members)} files)")

if "--clef" in sys.argv:
    out = ROOT / "models/clef-flash-4bit"
    snapshot_download("mlx-community/clef-flash-4bit", local_dir=out)
    print("ok", out.relative_to(ROOT))
