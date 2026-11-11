"""Pothole detector: finds and boxes each pothole (Faster R-CNN, MobileNetV3-Large FPN, torchvision).

Trained by scripts/train_detector.py on RDD2022 India vehicle-camera photos.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Half-precision copy (38 MB) of the trained weights; scripts/train_detector.py writes the full .pth.
WEIGHTS = ROOT / "models/det-frcnn-mnv3.fp16.pth"
SIZE = 720


def build_model(pretrained=True):
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None,
                                              min_size=SIZE, max_size=SIZE, box_score_thresh=0.01)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


def load(device="cpu"):
    """Return detect(img) -> list of (score, [x1, y1, x2, y2]) in the image's own pixels, best first."""
    import torch
    from torchvision.transforms import functional as F

    model = build_model(pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.to(device).eval()

    @torch.no_grad()
    def detect(img):
        out = model([F.to_tensor(img).to(device)])[0]
        return [(float(s), [round(float(v), 1) for v in b]) for s, b in zip(out["scores"].cpu(), out["boxes"].cpu())]

    return detect
