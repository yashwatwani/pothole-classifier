"""Is there a pothole? Runs fully offline on CPU.

  python predict.py photo.jpg                       # phone photo (handheld: classifier)
  python predict.py folder/ --json
  python predict.py frame.jpg --camera vehicle      # vehicle: classifier + detector, with boxes
  python predict.py dashcam.mp4                     # video: vehicle mode, 2 frames/s, pothole events
  python predict.py dashcam.mp4 --draw out/         # also save each event's best frame with boxes drawn
Cutoffs come from models/v4-mnv3-384.thresholds.json (handheld) and models/vehicle.json (vehicle).
"""
import argparse, json, sys, time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from pothole.predictors import load_camera_models

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def frames(path, fps):
    """Yield (seconds, PIL image) sampled at roughly `fps` frames per second."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open video {path}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()


def draw(img, result, out_path):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for b in result["boxes"]:
        d.rectangle(b["box"], outline=(255, 200, 0), width=max(2, img.width // 240))
        d.text((b["box"][0] + 3, b["box"][1] + 2), f"{b['score']:.0%}", fill=(255, 200, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


def video_events(path, analyze, fps, gap=1.5):
    """Group pothole frames less than `gap` seconds apart into one event; keep each event's best frame."""
    events, n = [], 0
    for t, img in frames(path, fps):
        n += 1
        r = analyze(img, "vehicle")
        if not r["pothole"]:
            continue
        p = r["probability"]
        if events and t - events[-1]["end_s"] <= gap:
            e = events[-1]
            e["end_s"], e["frames"] = t, e["frames"] + 1
            if p > e["peak_probability"]:
                e.update(peak_probability=p, peak_s=t, boxes=r["boxes"], _img=img, _r=r)
        else:
            events.append({"start_s": t, "end_s": t, "peak_s": t, "peak_probability": p, "frames": 1,
                           "boxes": r["boxes"], "_img": img, "_r": r})
    return events, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--camera", choices=["handheld", "vehicle"], help="default: vehicle for videos, handheld for photos")
    ap.add_argument("--fps", type=float, default=2, help="frames per second to check in videos")
    ap.add_argument("--draw", type=Path, metavar="DIR", help="save photos / event frames with pothole boxes drawn")
    ap.add_argument("--json", action="store_true", help="one JSON object per line")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS | VIDEO_EXTS) if p.is_dir() else [p]
    if not files:
        sys.exit("no images or videos found")

    analyze = load_camera_models()
    for f in files:
        if f.suffix.lower() in VIDEO_EXTS:
            t0 = time.perf_counter()
            events, n = video_events(f, analyze, a.fps)
            secs = time.perf_counter() - t0
            if not a.json:
                print(f"{f}: {len(events)} pothole event(s) in {n} frames checked ({secs:.1f} s)")
            for k, e in enumerate(events, 1):
                img, r = e.pop("_img"), e.pop("_r")
                if a.draw:
                    draw(img, r, a.draw / f"{f.stem}_event{k}_{e['peak_s']:.1f}s.jpg")
                if a.json:
                    print(json.dumps({"file": str(f), "camera": "vehicle", **{k2: round(v, 3) if isinstance(v, float) else v for k2, v in e.items()}}))
                else:
                    print(f"  {e['start_s']:7.1f}s - {e['end_s']:7.1f}s  peak {e['peak_probability']:.0%} at {e['peak_s']:.1f}s, {len(e['boxes'])} box(es)")
            continue
        camera = a.camera or "handheld"
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        t0 = time.perf_counter()
        r = analyze(img, camera)
        ms = (time.perf_counter() - t0) * 1000
        if a.draw and r["boxes"]:
            draw(img, r, a.draw / f"{f.stem}_boxes.jpg")
        if a.json:
            print(json.dumps({"file": str(f), **r, "ms": round(ms, 1)}))
        else:
            boxes = f"  {len(r['boxes'])} box(es)" if camera == "vehicle" else ""
            print(f"{'POTHOLE ' if r['pothole'] else 'no pothole'}  {r['probability']:6.1%}  {ms:5.1f} ms{boxes}  {f}")


if __name__ == "__main__":
    main()
