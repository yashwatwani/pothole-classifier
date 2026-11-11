"""Local HTTP API for the app.

  uvicorn serve:app --port 8000
  curl -F file=@photo.jpg localhost:8000/predict                    # handheld phone photo
  curl -F file=@frame.jpg "localhost:8000/predict?camera=vehicle"   # vehicle camera frame
"""
import io, time

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from pothole.predictors import cutoffs, load_ours

app = FastAPI(title="Pothole classifier")
predict, _, notes = load_ours()
CUTOFFS = cutoffs()


@app.get("/health")
def health():
    return {"ok": True, "model": notes}


@app.post("/predict")
async def classify(file: UploadFile = File(...), camera: Literal["handheld", "vehicle"] = "handheld"):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    t0 = time.perf_counter()
    p = predict(img)
    return {"pothole": p >= CUTOFFS[camera], "probability": round(p, 4), "camera": camera,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}
