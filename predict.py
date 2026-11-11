"""Is there a pothole? Runs fully offline on CPU.

  python predict.py photo.jpg                       # phone photo (handheld cutoff)
  python predict.py folder/ --json
  python predict.py dashcam.mp4                     # video: vehicle cutoff, 2 frames/s
  python predict.py dashcam.mp4 --fps 4 --json      # one JSON line per pothole event
The cutoff per camera type comes from models/<model>.thresholds.json; --threshold overrides it.
"""
import argparse, json, sys, time
from pathlib import Path

from PIL import Image, ImageOps

from pothole.predictors import cutoffs, load_ours

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


def video_events(path, predict, threshold, fps, gap=1.5):
    """Group pothole frames that are less than `gap` seconds apart into one event."""
    events, n = [], 0
    for t, img in frames(path, fps):
        n += 1
        p = predict(img)
        if p < threshold:
            continue
        if events and t - events[-1]["end_s"] <= gap:
            e = events[-1]
            e["end_s"], e["frames"] = t, e["frames"] + 1
            if p > e["peak_probability"]:
                e["peak_probability"], e["peak_s"] = p, t
        else:
            events.append({"start_s": t, "end_s": t, "peak_s": t, "peak_probability": p, "frames": 1})
    return events, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--camera", choices=["handheld", "vehicle"], help="default: vehicle for videos, handheld for photos")
    ap.add_argument("--threshold", type=float, help="override the tuned cutoff")
    ap.add_argument("--fps", type=float, default=2, help="frames per second to check in videos")
    ap.add_argument("--json", action="store_true", help="one JSON object per line")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS | VIDEO_EXTS) if p.is_dir() else [p]
    if not files:
        sys.exit("no images or videos found")

    predict, _, _ = load_ours()
    cut = cutoffs()
    for f in files:
        is_video = f.suffix.lower() in VIDEO_EXTS
        camera = a.camera or ("vehicle" if is_video else "handheld")
        threshold = a.threshold if a.threshold is not None else cut[camera]
        if is_video:
            t0 = time.perf_counter()
            events, n = video_events(f, predict, threshold, a.fps)
            secs = time.perf_counter() - t0
            if a.json:
                for e in events:
                    print(json.dumps({"file": str(f), "camera": camera, **{k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()}}))
            else:
                print(f"{f}: {len(events)} pothole event(s) in {n} frames checked ({secs:.1f} s, cutoff {threshold:.3f}, {camera})")
                for e in events:
                    print(f"  {e['start_s']:7.1f}s - {e['end_s']:7.1f}s  peak {e['peak_probability']:.0%} at {e['peak_s']:.1f}s")
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        t0 = time.perf_counter()
        p = predict(img)
        ms = (time.perf_counter() - t0) * 1000
        if a.json:
            print(json.dumps({"file": str(f), "camera": camera, "pothole": p >= threshold, "probability": round(p, 4), "ms": round(ms, 1)}))
        else:
            print(f"{'POTHOLE ' if p >= threshold else 'no pothole'}  {p:6.1%}  {ms:5.1f} ms  {f}")


if __name__ == "__main__":
    main()
