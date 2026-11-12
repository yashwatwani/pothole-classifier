# Pothole classifier

One job: look at a road photo or video frame and say **pothole / no pothole**.
Open data, open-source models, runs fully offline on a laptop CPU. No image ever leaves the machine.
Works for **handheld phone photos** (classifier, ~7.5 ms per image) and **vehicle cameras** (dashcam or a phone mounted
in a car, including video): there a box detector also runs (~80 ms per frame) and marks where each pothole is.

## Results (held-out test set, 1,770 images, never seen in training)

AUC compares models without picking a cutoff: 0.5 is a coin flip, 1.0 is perfect.

| Model | All photos AUC | Indian vehicle camera: AUC / accuracy | Speed per image |
|---|---|---|---|
| **Ours v2: MobileNetV3, 384 px, trained with Indian data** (default) | **0.927** | **0.872 / 89.3%** | 7.5 ms, CPU |
| Ours: box detector alone (Faster R-CNN MobileNetV3, trained on Indian boxes only) | 0.559 | 0.859 / 89.8% | 80 ms, CPU |
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

### Pothole boxes (vehicle cameras)

The classifier judges a whole frame; on Indian vehicle footage it partly learns "what a bad road scene looks like".
So vehicle mode also runs a detector that draws a box around each pothole: Faster R-CNN with a MobileNetV3-Large FPN
backbone (torchvision, BSD licence, COCO-pretrained), trained on RDD2022 India's pothole boxes at 720 px
(`scripts/train_detector.py`, 14 epochs, ~55 min on an M5 Pro).

On the 1,163 Indian test photos (201 potholes), `scripts/eval_detector.py`:

| Box score cutoff | Potholes found with a correctly placed box (IoU ≥ 0.5) | Small / medium / large | Boxes per 100 pothole-free photos |
|---|---|---|---|
| 0.3 | 90 / 201 (45%) | 43% / 42% / 49% | 9.7 |
| 0.5 | 78 / 201 (39%) | 35% / 38% / 42% | 5.2 |
| 0.7 | 65 / 201 (32%) | 25% / 32% / 37% | 3.6 |

Box AP@0.5: 0.36 on test, 0.47 on validation. It only knows Indian vehicle footage: on handheld close-ups and the
South African dashcam set it is at chance, because those datasets have no boxes to train on.

### Camera modes and cutoffs

| Camera | What runs | Cutoff (chosen on validation) | On test: potholes caught | Alerts that are real | False alarms per 100 normal photos |
|---|---|---|---|---|---|
| Handheld (someone photographs a pothole) | classifier | 0.050, to catch ~90% | 81% | 84% | 30 |
| Vehicle (dashcam / mounted phone) | max(classifier, detector) + boxes | 0.400, to catch ~85% | 85% | 75% | 8.2 |
| Vehicle, Indian footage only | same | same | 68% (classifier alone: 59%) | 51% | 9.5 |

Handheld cutoffs come from `scripts/tune_threshold.py` (`models/v2-mnv3-384.thresholds.json`), the vehicle cutoff from
`scripts/combine_vehicle.py` (`models/vehicle.json`). "Pothole if either model is confident" beat both models alone and
a 50/50 average across all vehicle photos (AUC 0.949 vs 0.943 classifier only). The handheld cutoff is aggressive on
purpose: someone reporting a pothole is worse to reject than a false alarm a reviewer dismisses.

The trade-off is still steep on Indian vehicle footage: catching 90% of potholes there means flagging about 1 in 3 normal frames.

### Robustness (classifier AUC, `scripts/robustness.py`)

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

.venv/bin/python predict.py photo.jpg                     # phone photo (handheld); a folder works too
.venv/bin/python predict.py frame.jpg --camera vehicle    # vehicle: classifier + detector, prints pothole boxes
.venv/bin/python predict.py dashcam.mp4 --draw out/       # video: 2 frames/s, pothole events with timestamps; saves each event's frame with boxes
.venv/bin/uvicorn serve:app --port 8000
curl -F file=@frame.jpg "localhost:8000/predict?camera=vehicle"
# {"pothole": true, "probability": 0.98, "camera": "vehicle", "boxes": [{"score": 0.98, "box": [177.1, 415.9, 238.6, 510.2]}], "ms": 125}
```

Rebuild from scratch:

```bash
.venv/bin/python scripts/download_data.py         # ~900 MB into data/raw/ (add --clef for the 6.2 GB Clef-flash weights)
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train.py --model mobilenetv3_large_100 --size 384 --epochs 12 --name v2-mnv3-384
.venv/bin/python scripts/eval.py v2-384 --split val && .venv/bin/python scripts/eval.py v2-384
.venv/bin/python scripts/tune_threshold.py v2-384 --recall-handheld 0.90 --recall-vehicle 0.85
.venv/bin/python scripts/robustness.py v2-384
.venv/bin/python scripts/train_detector.py --epochs 14   # then save a half-precision copy as models/det-frcnn-mnv3.fp16.pth
.venv/bin/python scripts/eval_detector.py
.venv/bin/python scripts/eval.py detector --split val && .venv/bin/python scripts/eval.py detector
.venv/bin/python scripts/combine_vehicle.py --recall 0.85  # writes models/vehicle.json
.venv/bin/python scripts/eval.py v1 siglip-zeroshot hf-resnet50
.venv-mlx/bin/python scripts/eval_clef.py         # zero-shot Clef-flash; env setup in requirements-clef.txt
.venv/bin/python scripts/compare.py               # writes results/RESULTS.md
```

## Layout

```
predict.py, serve.py        the harness: CLI (photos, folders, video) + local HTTP API
pothole/metrics.py          eval core (accuracy, P/R/F1, AUC, latency, per source and camera type, every score)
pothole/predictors.py       every model behind one predict(image) -> P(pothole) interface; load_camera_models() is what predict.py/serve.py run
pothole/detector.py         the pothole box detector
scripts/                    download_data, prepare_data, train, train_detector, eval, eval_detector, eval_clef,
                            tune_threshold, combine_vehicle, robustness, compare, gallery
models/                     classifier .onnx files, detector det-frcnn-mnv3.fp16.pth (38 MB), cutoffs (*.thresholds.json, vehicle.json)
results/                    eval outputs
```

## Known limits

- **Night and low light** are the main weakness: from a vehicle in the dark it catches under half the potholes. It needs night-time training photos.
- **Small, distant potholes from vehicles.** Even with the box detector, about 6 in 10 Indian potholes don't get a correctly placed box, and vehicle mode catches 68% of Indian potholes at the default cutoff. More box-labelled data is the next lever; training longer (24 epochs) scored lower on validation (AP 0.450 vs 0.465), so the 14-epoch detector is kept.
- **The detector only knows Indian vehicle footage.** Handheld mode doesn't use it, and it adds nothing on other countries' dashcams yet.
- **Heavy video compression** costs accuracy (vehicle AUC 0.79 at JPEG quality 15). Upload frames at a reasonable quality.
- **Few handheld test photos** (89), so handheld numbers are less certain than vehicle ones.
- Off-the-shelf models are compared at their default cutoff; compare them on AUC.
- A YOLO pothole detector was left out: its weights ship as a pickle (`best.pt`), which can run code when loaded, and it is AGPL-licensed.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The datasets keep their own licences, listed under Data above. Models trained on RDD2022 data should credit it (CC BY-SA 4.0).
