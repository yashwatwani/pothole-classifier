# Pothole classifier

One job: look at a road photo and say **pothole / no pothole**.
Open data, open-source model, runs fully offline on a laptop CPU (~3–4 ms per image). No image ever leaves the machine.

## Results (held-out test set, 607 images)

| Model | Accuracy | Recall (potholes caught) | F1 | Speed per image |
|---|---|---|---|---|
| **Ours: MobileNetV3, fine-tuned** (default) | **89.5%** | **84.5%** | **0.885** | **3.4 ms, CPU** |
| Ours: EfficientNet-B0, fine-tuned | 88.1% | 85.2% | 0.873 | 5.6 ms, CPU |
| Off-the-shelf HF classifier (`punchnami/resnet50-pothole-classification`) | 65.4% | 35.4% | 0.495 | 13 ms, GPU |
| Zero-shot SigLIP (`google/siglip-base-patch16-224`), no training | 58.2% | 13.1% | 0.230 | 31 ms, GPU |

Full table and per-source scores: [results/RESULTS.md](results/RESULTS.md). Clef-flash (zero-shot, Jev-compatible) is downloaded and scripted but not yet scored.
Each `results/<model>.json` also lists every image the model got wrong.

## Data

Three open datasets from Hugging Face, merged into one binary set by `scripts/prepare_data.py`:

| Source | What | Licence |
|---|---|---|
| [mabughali/miia-pothole-train](https://huggingface.co/datasets/mabughali/miia-pothole-train) | 4,026 dashcam images (South Africa) | Apache-2.0 |
| [taroii/pothole-detection](https://huggingface.co/datasets/taroii/pothole-detection) | 437 images, pre-split | not stated |
| [Arpitraj01/Pothole_classification](https://huggingface.co/datasets/Arpitraj01/Pothole_classification) | ~490 images from 50 km of Kerala roads, severity labels → binary | MIT |

Near-duplicate photos are removed with a perceptual hash (1,365 MIIA dashcam frames were near-copies), so no photo sits in both train and test. Final split: 2,172 train / 704 val / 607 test.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision timm onnx onnxruntime pillow pyarrow huggingface_hub scikit-learn imagehash fastapi uvicorn python-multipart transformers

.venv/bin/python predict.py photo.jpg             # or a folder; --json for machine output
.venv/bin/uvicorn serve:app --port 8000           # POST an image to /predict
curl -F file=@photo.jpg localhost:8000/predict    # {"pothole": true, "probability": 0.9978, "ms": 4.4}
```

Rebuild from scratch:

```bash
.venv/bin/python scripts/prepare_data.py          # download sources into data/raw first (see script docstring)
.venv/bin/python scripts/train.py --model mobilenetv3_large_100 --epochs 10
.venv/bin/python scripts/eval.py ours siglip-zeroshot hf-resnet50
.venv-mlx/bin/python scripts/eval_clef.py         # zero-shot Clef-flash, needs models/clef-flash-4bit
.venv/bin/python scripts/compare.py               # writes results/RESULTS.md
```

## Layout

```
predict.py, serve.py        the harness: CLI + local HTTP API
pothole/metrics.py          eval core (accuracy, P/R/F1, AUC, latency, per-source, mistakes)
pothole/predictors.py       every model behind one predict(image) -> P(pothole) interface
scripts/                    prepare_data, train, eval, eval_clef, compare
models/*.onnx               trained weights (in git; .pt checkpoints are not)
results/                    eval outputs
```

## Known limits

- Recall is ~85%: about 1 in 7 real potholes is missed at threshold 0.5. Lower `--threshold` to catch more, at the cost of more false alarms.
- Weakest on Indian "normal" roads: the Kerala set has only 49 no-pothole photos, and broken edges, patches and shadows get flagged. **The fastest win is adding a few hundred photos from our own app, labelled, and retraining (about 2 minutes on this Mac).**
- The taroii source scores lowest (F1 0.77); its images are small and low-resolution.
- Off-the-shelf models used their default threshold, so compare them on AUC too. Ours is still best there (0.949 vs 0.872 and 0.695).
- A YOLO pothole detector was left out: its weights ship as a pickle (`best.pt`), which can run code when loaded, and it is AGPL-licensed.
