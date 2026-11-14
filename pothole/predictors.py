"""Every model we compare, behind one interface: load_<name>() -> (predict, device, notes)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Default model: trained with Indian and Japanese vehicle-camera photos (RDD2022), 384 px input.
DEFAULT_MODEL, DEFAULT_SIZE = "v3-mnv3-384", 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cutoffs(model=DEFAULT_MODEL):
    """Per-camera pothole cutoffs chosen on the validation split (scripts/tune_threshold.py)."""
    import json

    f = ROOT / "models" / f"{model}.thresholds.json"
    return json.loads(f.read_text())["cutoffs"] if f.exists() else {"handheld": 0.5, "vehicle": 0.5}


def load_ours(model=DEFAULT_MODEL, size=DEFAULT_SIZE):
    """Our fine-tuned model, exported to ONNX, on CPU (how it would be deployed)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ROOT / "models" / f"{model}.onnx"), providers=["CPUExecutionProvider"])

    def predict(img):
        x = (np.asarray(img.resize((size, size)), dtype=np.float32) / 255 - MEAN) / STD
        logits = sess.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    return predict, "cpu (onnxruntime)", f"Ours: {model} fine-tuned on our train split"


def load_siglip(repo="google/siglip-base-patch16-224"):
    """Zero-shot: no pothole training at all, just text prompts."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(repo, use_safetensors=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(repo)
    prompts = {
        1: ["a photo of a road with a pothole", "a damaged road with potholes", "a pothole filled with water"],
        0: ["a photo of a smooth road", "a road with no potholes", "a normal street"],
    }
    texts = prompts[0] + prompts[1]
    t = proc(text=texts, padding="max_length", return_tensors="pt").to(device)

    def predict(img):
        with torch.no_grad():
            px = proc(images=img, return_tensors="pt").to(device)
            logits = model(**t, **px).logits_per_image[0]
        probs = logits.softmax(-1).cpu().numpy()
        return float(probs[len(prompts[0]):].sum())

    return predict, device, f"Zero-shot {repo}, prompt ensemble, no training"


def load_hf_resnet50(repo="punchnami/resnet50-pothole-classification"):
    """An existing open pothole classifier from Hugging Face, used as-is."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(repo, use_safetensors=True).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(repo)
    idx = model.config.label2id["pothole"]

    def predict(img):
        with torch.no_grad():
            logits = model(**proc(images=img, return_tensors="pt").to(device)).logits[0]
        return float(logits.softmax(-1)[int(idx)])

    return predict, device, f"Off-the-shelf {repo} (training data unknown, may overlap our test set)"


def load_yolo(repo="keremberke/yolov8n-pothole-segmentation"):
    """An existing open pothole *detector*; P(pothole) = highest box confidence."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    model = YOLO(hf_hub_download(repo, "best.pt"))

    def predict(img):
        boxes = model.predict(img, verbose=False, conf=0.01)[0].boxes
        return float(boxes.conf.max()) if len(boxes) else 0.0

    return predict, "auto (ultralytics)", f"Off-the-shelf detector {repo}, AGPL-3.0 (ultralytics)"


def load_detector(device="cpu"):
    """Box detector used as a photo classifier: P(pothole) = its most confident box (0 if none)."""
    from pothole import detector

    detect = detector.load(device)

    def predict(img):
        boxes = detect(img)
        return boxes[0][0] if boxes else 0.0

    return predict, device, "Ours: Faster R-CNN MobileNetV3 detector, trained on RDD2022 India and Japan boxes; photo score = top box"


def load_camera_models():
    """What predict.py and serve.py run. Returns analyze(img, camera) -> dict.

    handheld: the classifier with its handheld cutoff.
    vehicle:  max(classifier, detector) with the cutoff in models/vehicle.json, plus the boxes
              the detector drew at that cutoff. Falls back to the classifier alone if the
              detector weights or models/vehicle.json are missing.
    """
    import json

    from pothole import detector

    classify, _, _ = load_ours()
    cut = cutoffs()
    vehicle_cfg = ROOT / "models/vehicle.json"
    detect = None
    if detector.WEIGHTS.exists() and vehicle_cfg.exists():
        detect = detector.load()
        cut = {**cut, "vehicle": json.loads(vehicle_cfg.read_text())["cutoff"]}

    def analyze(img, camera="handheld"):
        p = classify(img)
        boxes = []
        if camera == "vehicle" and detect:
            found = detect(img)
            p = max(p, found[0][0] if found else 0.0)
            boxes = [{"score": round(s, 3), "box": b} for s, b in found if s >= cut["vehicle"]]
        return {"pothole": p >= cut[camera], "probability": round(p, 4), "camera": camera, "boxes": boxes}

    return analyze


LOADERS = {
    "v1": lambda: load_ours("mobilenetv3_large_100", 224),  # original model, before Indian data
    "v2-224": lambda: load_ours("v2-mnv3-224", 224),
    "v2-384": lambda: load_ours("v2-mnv3-384", 384),
    "v3-384": lambda: load_ours("v3-mnv3-384", 384),  # + RDD2022 Japan
    "v1-effnet": lambda: load_ours("efficientnet_b0", 256),
    "detector": load_detector,
    "siglip-zeroshot": load_siglip,
    "hf-resnet50": load_hf_resnet50,
    "yolov8n-pothole": load_yolo,
}
