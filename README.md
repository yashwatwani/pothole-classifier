# Pothole classifier

One job: look at a road photo or video frame and say **pothole / no pothole**.
Open data, open-source model, runs fully offline on a laptop CPU (~7.5 ms per image). No image ever leaves the machine.
Works for **handheld phone photos** and **vehicle cameras** (dashcam or a phone mounted in a car), including video.

## Results (held-out test set, 1,770 images, never seen in training)

AUC compares models without picking a cutoff: 0.5 is a coin flip, 1.0 is perfect.

| Model | All photos AUC | Indian vehicle camera: AUC / accuracy | Speed per image |
|---|---|---|---|
| **Ours v2: MobileNetV3, 384 px, trained with Indian data** (default) | **0.927** | **0.872 / 89.3%** | 7.5 ms, CPU |
| Ours v2, 224 px | 0.908 | 0.834 / 87.7% | 3.9 ms, CPU |
| Ours v1: original model (no Indian data, 224 px) | 0.647 | 0.472 / 27.3% | 3.4 ms, CPU |
| Ours v1, EfficientNet-B0 | 0.715 | see RESULTS.md | 7.2 ms, CPU |
| Zero-shot Clef-flash 9B (Cloudflare, Jev-compatible, 4-bit MLX) | 0.623 | 0.477 / 83.1% (catches only 3% of potholes) | ~515 ms, GPU |
| Zero-shot SigLIP (`google/siglip-base-patch16-224`) | 0.665 | see RESULTS.md | 32 ms, GPU |
| Off-the-shelf HF classifier (`punchnami/resnet50-pothole-classification`) | 0.644 | see RESULTS.md | 14 ms, GPU |

Full tables by camera type and by source: [results/RESULTS.md](results/RESULTS.md). Each `results/<model>.json` lists every photo's score and every mistake.

### What changed between v1 and v2

v1 scored 89.5% accuracy on its original 607-photo test set, but that set had only 17 Indian photos. On 1,163 Indian vehicle-camera photos it was no better than chance (AUC 0.47): it flagged almost every Indian road as a pothole. From a vehicle, an Indian pothole is small: the typical one covers ~1.5% of the frame.

v2 fixes most of that:
- trained with 7,706 Indian photos from RDD2022
- gentler random crops, so small potholes aren't cut out of "pothole" photos during training
- random rotations and blur, for sideways and shaky phone photos
- 384 px input instead of 224

### Camera modes and cutoffs

Each camera type gets its own alert cutoff, chosen on the validation split (`scripts/tune_threshold.py`) and stored in `models/v2-mnv3-384.thresholds.json`.

| Camera | Target | On test: potholes caught | Alerts that are real | False alarms per 100 normal photos |
|---|---|---|---|---|
| Handheld (someone photographs a pothole) | catch ~90% | 81% | 84% | 30 |
| Vehicle (dashcam / mounted phone) | catch ~85% | 82% | 75% | 8 |

The trade-off is steep on Indian vehicle footage: catching 90% of potholes there means flagging about 1 in 3 normal frames.

### Robustness (AUC, `scripts/robustness.py`)

| Condition | Handheld v1 → v2 | Vehicle v1 → v2 |
|---|---|---|
| Clean | 0.83 → 0.87 | 0.65 → 0.94 |
| Motion blur | 0.85 → 0.82 | 0.67 → 0.91 |
| Low resolution (1/4) | 0.68 → 0.89 | 0.69 → 0.90 |
| Heavy JPEG (q=15) | 0.86 → 0.88 | 0.68 → 0.79 |
| Rotated 90° | 0.83 → 0.88 | not applicable |
| **Low light + noise** | 0.74 → **0.64** | 0.65 → **0.71** |

## Data

Four open datasets, merged into one binary set by `scripts/prepare_data.py`:

| Source | Camera | What | Licence |
|---|---|---|---|
| [mabughali/miia-pothole-train](https://huggingface.co/datasets/mabughali/miia-pothole-train) | vehicle | 4,026 dashcam images (South Africa) | Apache-2.0 |
| [aiswarya9302/RDD_2022India](https://huggingface.co/datasets/aiswarya9302/RDD_2022India) | vehicle | 7,706 images from RDD2022 India (Arya et al. 2022). Pothole = any D40 box; crack-only and undamaged photos count as normal. | CC BY-SA 4.0 |
| [taroii/pothole-detection](https://huggingface.co/datasets/taroii/pothole-detection) | handheld | 437 images, pre-split | not stated |
| [Arpitraj01/Pothole_classification](https://huggingface.co/datasets/Arpitraj01/Pothole_classification) | handheld | ~490 images from 50 km of Kerala roads, severity labels → binary | MIT |

Near-duplicate photos are removed with a perceptual hash (1,365 MIIA and 86 RDD frames were near-copies), so no photo sits in both train and test. Final split: 7,459 train / 1,874 val / 1,770 test.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python predict.py photo.jpg                     # phone photo (handheld cutoff); a folder works too
.venv/bin/python predict.py dashcam.mp4                   # video: vehicle cutoff, 2 frames/s, pothole events with timestamps
.venv/bin/python predict.py frame.jpg --camera vehicle --json
.venv/bin/uvicorn serve:app --port 8000
curl -F file=@photo.jpg "localhost:8000/predict?camera=handheld"   # {"pothole": true, "probability": 0.95, "camera": "handheld", "ms": 8.1}
```

Rebuild from scratch:

```bash
.venv/bin/python scripts/download_data.py         # ~900 MB into data/raw/ (add --clef for the 6.2 GB Clef-flash weights)
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train.py --model mobilenetv3_large_100 --size 384 --epochs 12 --name v2-mnv3-384
.venv/bin/python scripts/eval.py v2-384 --split val && .venv/bin/python scripts/eval.py v2-384
.venv/bin/python scripts/tune_threshold.py v2-384 --recall-handheld 0.90 --recall-vehicle 0.85
.venv/bin/python scripts/robustness.py v2-384
.venv/bin/python scripts/eval.py v1 siglip-zeroshot hf-resnet50
.venv-mlx/bin/python scripts/eval_clef.py         # zero-shot Clef-flash; env setup in requirements-clef.txt
.venv/bin/python scripts/compare.py               # writes results/RESULTS.md
```

## Layout

```
predict.py, serve.py        the harness: CLI (photos, folders, video) + local HTTP API
pothole/metrics.py          eval core (accuracy, P/R/F1, AUC, latency, per source and camera type, every score)
pothole/predictors.py       every model behind one predict(image) -> P(pothole) interface; default model and cutoffs
scripts/                    download_data, prepare_data, train, eval, eval_clef, tune_threshold, robustness, compare, gallery
models/*.onnx               trained weights (in git; .pt checkpoints are not)
results/                    eval outputs
```

## Known limits

- **Night and low light** are the main weakness: from a vehicle in the dark it catches under half the potholes. It needs night-time training photos.
- **Small, distant potholes from vehicles.** The model judges the whole frame. A model that boxes each pothole would do better; RDD2022 already has the boxes.
- **Heavy video compression** costs accuracy (vehicle AUC 0.79 at JPEG quality 15). Upload frames at a reasonable quality.
- **Few handheld test photos** (89), so handheld numbers are less certain than vehicle ones.
- Off-the-shelf models are compared at their default cutoff; compare them on AUC.
- A YOLO pothole detector was left out: its weights ship as a pickle (`best.pt`), which can run code when loaded, and it is AGPL-licensed.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The datasets keep their own licences, listed under Data above. Models trained on RDD2022 data should credit it (CC BY-SA 4.0).
