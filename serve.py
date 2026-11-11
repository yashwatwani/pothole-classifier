"""Local HTTP API for the app.

  uvicorn serve:app --port 8000
  curl -F file=@photo.jpg localhost:8000/predict                    # handheld phone photo
  curl -F file=@frame.jpg "localhost:8000/predict?camera=vehicle"   # vehicle camera frame, with pothole boxes
"""
import io, time

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from pothole.predictors import DEFAULT_MODEL, load_camera_models

app = FastAPI(title="Pothole classifier")
analyze = load_camera_models()


@app.get("/health")
def health():
    return {"ok": True, "handheld": f"classifier {DEFAULT_MODEL}", "vehicle": "max(classifier, detector), with boxes"}


@app.post("/predict")
async def classify(file: UploadFile = File(...), camera: Literal["handheld", "vehicle"] = "handheld"):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    t0 = time.perf_counter()
    result = analyze(img, camera)
    return {**result, "ms": round((time.perf_counter() - t0) * 1000, 1)}
