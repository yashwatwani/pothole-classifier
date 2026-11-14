"""Download the open datasets into data/raw/ (about 4 GB).

  python scripts/download_data.py
Then run scripts/prepare_data.py. Add --clef to also fetch the 6.2 GB Clef-flash 4-bit
weights into models/clef-flash-4bit for scripts/eval_clef.py.
"""
import hashlib, io, json, shutil, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
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


def get(url, headers=None, **kw):
    # Mendeley refuses Python's default client name
    headers = {"User-Agent": "pothole-classifier-downloader/1.0", **(headers or {})}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), **kw)


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, so zipfile can pull single members out of it."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with get(url, headers={"Range": "bytes=0-0"}) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        with get(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data


def fetch(url, dest, sha256):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        for attempt in range(4):  # Mendeley drops the odd connection
            try:
                with get(url, timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(5)
        tmp.rename(dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        dest.unlink()
        raise RuntimeError(f"checksum mismatch: {dest}")


def mendeley(dataset, out, keep, by_id=False):
    """Files of a public Mendeley Data set whose folder path starts with one of `keep`.
    by_id names files by their id, for sets whose file names clash on a case-insensitive disk."""
    api = f"https://data.mendeley.com/public-api/datasets/{dataset}"
    meta = json.load(get(api))
    folders = {f["id"]: f for f in json.load(get(f"{api}/folders/{meta['version']}"))}

    def path(fid):
        parts = []
        while fid:
            parts.append(folders[fid]["name"]); fid = folders[fid].get("parent_id")
        return "/".join(reversed(parts))

    jobs = []
    for f in meta["files"]:
        folder = path(f["folder_id"])
        if folder.startswith(tuple(keep)):
            name = f["id"] + Path(f["filename"]).suffix.lower() if by_id else f["filename"]
            jobs.append((f["content_details"]["download_url"], out / folder / name, f["content_details"]["sha256_hash"]))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda j: fetch(*j), jobs))
    print("ok", out.relative_to(ROOT), f"({len(jobs)} files)")


# RDD2022 Japan (Arya et al. 2022; CC BY-SA 4.0 per the paper, CC BY 4.0 on FigShare): Pascal VOC labels,
# D40 = pothole; only the train part is labelled. The FigShare release is one 13 GB zip of per-country
# zips, so only Japan's is read out of it.
zip_path = ROOT / "data/raw/rdd2022_world_zip/Japan.zip"
if not zip_path.exists():
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    rdd = zipfile.ZipFile(RangeFile("https://ndownloader.figshare.com/files/38030910"))
    with rdd.open("RDD2022/Japan.zip") as src, open(zip_path.with_suffix(".part"), "wb") as dst:
        shutil.copyfileobj(src, dst, 16 << 20)
    zip_path.with_suffix(".part").rename(zip_path)
out = ROOT / "data/raw/rdd2022_world"
with zipfile.ZipFile(zip_path) as z:
    members = [m for m in z.namelist() if "/train/" in m and ".." not in m and not m.startswith("/")]
    z.extractall(out, members)
print("ok", out.relative_to(ROOT), f"Japan ({len(members)} files)")

# BharatPotHole (Kaggle, CC BY-SA 4.0): Indian dashcam frames with pothole boxes. Kaggle needs a login, so
# download it yourself from https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole (version 2)
# and save the zip as data/raw/bharatpothole_zip/archive.zip.
zip_path = ROOT / "data/raw/bharatpothole_zip/archive.zip"
if zip_path.exists():
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if ".." not in m and not m.startswith("/")]
        z.extractall(ROOT / "data/raw/bharatpothole", members)
    print("ok data/raw/bharatpothole", f"({len(members)} files)")
else:
    print(f"missing {zip_path.relative_to(ROOT)}: download BharatPotHole from Kaggle (see the comment above)")

# QR4Change (CC BY 4.0): pothole photos. Only the phone photos taken in Pune are used, and only for
# testing (see scripts/prepare_data.py). The pothole/no folder is mixed-source and not used.
mendeley("zndzygc3p3", ROOT / "data/raw/qr4change_pune", ["pothole/yes"], by_id=True)

if "--clef" in sys.argv:
    out = ROOT / "models/clef-flash-4bit"
    snapshot_download("mlx-community/clef-flash-4bit", local_dir=out)
    print("ok", out.relative_to(ROOT))
