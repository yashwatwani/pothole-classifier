"""Local HTTP API for the app.

  uvicorn serve:app --port 8000
  curl -F file=@photo.jpg localhost:8000/predict
"""
import io, time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from pothole.predictors import load_ours

THRESHOLD = 0.5
app = FastAPI(title="Pothole classifier")
predict, _, notes = load_ours()


@app.get("/health")
def health():
    return {"ok": True, "model": notes}


@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    t0 = time.perf_counter()
    p = predict(img)
    return {"pothole": p >= THRESHOLD, "probability": round(p, 4), "ms": round((time.perf_counter() - t0) * 1000, 1)}
