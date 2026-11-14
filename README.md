# Pothole classifier

One job: look at a road photo or video frame and say **pothole / no pothole**.
Open data, open-source models, runs fully offline on a laptop CPU. No image ever leaves the machine.
Works for **handheld phone photos** (classifier, ~7 ms per image) and **vehicle cameras** (dashcam or a phone mounted
in a car, including video): there a box detector also runs (~90 ms per frame) and marks where each pothole is.

## Results (held-out test set, 3,369 images, never seen in training)

AUC compares models without picking a cutoff: 0.5 is a coin flip, 1.0 is perfect.

| Model | All photos AUC | Indian vehicle camera: AUC / accuracy | Japanese vehicle camera AUC | Speed per image |
|---|---|---|---|---|
| **Ours v3: MobileNetV3, 384 px, trained with Indian and Japanese data** (default) | **0.935** | **0.876 / 89.9%** | **0.939** | 7.3 ms, CPU |
| Ours: box detector alone (Faster R-CNN MobileNetV3, trained on Indian and Japanese boxes) | 0.715 | 0.861 / 89.1% | 0.943 | 89 ms, CPU |
| Ours v2: same, Indian data only | 0.834 | 0.872 / 89.3% | 0.612 | 7.5 ms, CPU |
| Ours v2, 224 px | 0.815 | 0.834 / 87.7% | 0.616 | 3.9 ms, CPU |
| Ours v1: original model (neither, 224 px) | 0.668 | 0.472 / 27.3% | 0.656 | 3.6 ms, CPU |
| Ours v1, EfficientNet-B0 | 0.677 | 0.440 / 45.2% | 0.604 | 6.6 ms, CPU |
| Zero-shot Clef-flash 9B (Cloudflare, Jev-compatible, 4-bit MLX)* | 0.623 | 0.477 / 83.1% (catches only 3% of potholes) | not scored | ~515 ms, GPU |
| Zero-shot SigLIP (`google/siglip-base-patch16-224`) | 0.601 | 0.420 / 87.2% | 0.650 | 32 ms, GPU |
| Off-the-shelf HF classifier (`punchnami/resnet50-pothole-classification`) | 0.622 | 0.549 / 75.4% | 0.718 | 11 ms, GPU |

\* Scored on the original 1,770-photo test set, before the Japanese photos were added (about 30 minutes to re-run).

On 389 phone photos of potholes taken on Pune streets (test only, see Data), v3 flags 99.7% at the handheld cutoff
(v2: 94.6%). The set has no pothole-free photos, so this measures catch rate, not false alarms.

Full tables by camera type and by source: [results/RESULTS.md](results/RESULTS.md). Each `results/<model>.json` lists every photo's score and every mistake.

### What changed between v1 and v2

v1 scored 89.5% accuracy on its original 607-photo test set, but that set had only 17 Indian photos. On 1,163 Indian vehicle-camera photos it was no better than chance (AUC 0.47): it flagged almost every Indian road as a pothole. From a vehicle, an Indian pothole is small: the typical one covers ~1.5% of the frame.

v2 fixes most of that:
- trained with 7,706 Indian photos from RDD2022
- gentler random crops, so small potholes aren't cut out of "pothole" photos during training
- random rotations and blur, for sideways and shaky phone photos
- 384 px input instead of 224

### What changed between v2 and v3

v3 is the same recipe with 10,506 more vehicle-camera photos from RDD2022 Japan. Japanese roads went from AUC 0.61
to 0.94; Indian roads stayed level or better (AUC 0.872 → 0.876); on the original 1,770 test photos AUC went
0.927 → 0.929. At the vehicle cutoff the classifier alone now catches 69% of Indian potholes (v2: 59%) with the
same share of alerts real (50%).

### Pothole boxes (vehicle cameras)

The classifier judges a whole frame; on Indian vehicle footage it partly learns "what a bad road scene looks like".
So vehicle mode also runs a detector that draws a box around each pothole: Faster R-CNN with a MobileNetV3-Large FPN
backbone (torchvision, BSD licence, COCO-pretrained), trained on RDD2022 India's and Japan's pothole boxes at 720 px
(`scripts/train_detector.py`, 14 epochs, ~2 h on an M5 Pro). Gradient clipping keeps training stable: without it,
one oversized update near the peak learning rate turned every weight into NaN.

`scripts/eval_detector.py`, potholes found with a correctly placed box (IoU ≥ 0.5):

| Box score cutoff | India: 1,163 photos, 201 potholes | Japan: 1,599 photos, 332 potholes | Boxes per 100 pothole-free photos (India / Japan) |
|---|---|---|---|
| 0.3 | 82 (41%) | 174 (52%) | 8.0 / 4.5 |
| 0.5 | 65 (32%) | 155 (47%) | 5.6 / 2.4 |
| 0.7 | 55 (27%) | 129 (39%) | 3.3 / 1.1 |

Box AP@0.5 on test: India 0.34 (the India-only detector: 0.35), Japan 0.49 (India-only: 0.00); 0.41 on validation.
Small, medium and large potholes are found at similar rates. It only knows Indian and Japanese vehicle footage: on
handheld close-ups and the South African dashcam set it is near chance, because those datasets have no boxes to train on.

### Camera modes and cutoffs

| Camera | What runs | Cutoff (chosen on validation) | On test: potholes caught | Alerts that are real | False alarms per 100 normal photos |
|---|---|---|---|---|---|
| Handheld (someone photographs a pothole) | classifier | 0.057, to catch ~90% | 88% | 80% | 43 (of only 30 normal photos) |
| Vehicle (dashcam / mounted phone) | max(classifier, detector) + boxes | 0.388, to catch ~85% | 86% | 73% | 6.8 |
| Vehicle, Indian footage only | same | same | 66% | 52% | 9.1 |
| Vehicle, Japanese footage only | same | same | 89% | 71% | 5.5 |

Handheld cutoffs come from `scripts/tune_threshold.py` (`models/v3-mnv3-384.thresholds.json`), the vehicle cutoff from
`scripts/combine_vehicle.py` (`models/vehicle.json`). "Pothole if either model is confident" ranks vehicle photos
better than the classifier alone (AUC 0.955 vs 0.943; India 0.887 vs 0.876; Japan 0.963 vs 0.939). The handheld cutoff
is aggressive on purpose: someone reporting a pothole is worse to reject than a false alarm a reviewer dismisses.

The trade-off is still steep on Indian vehicle footage: catching 90% of potholes there means flagging about 1 in 3
normal frames (36 per 100; on Japanese footage, 6 per 100).

### Robustness (classifier AUC, `scripts/robustness.py`)

| Condition | Handheld v2 → v3 | Vehicle v2 → v3 |
|---|---|---|
| Clean | 0.87 → 0.88 | 0.84 → 0.94 |
| Motion blur | 0.82 → 0.86 | 0.81 → 0.92 |
| Low resolution (1/4) | 0.89 → 0.89 | 0.80 → 0.92 |
| Heavy JPEG (q=15) | 0.88 → 0.84 | 0.74 → 0.87 |
| Rotated 90° | 0.88 → 0.85 | not applicable |
| **Low light + noise** | 0.64 → **0.77** | 0.66 → **0.73** |

On the current test set, which now includes Japanese vehicle photos.

## Data

Open datasets, merged into one binary set by `scripts/prepare_data.py`:

| Source | Camera | What | Licence |
|---|---|---|---|
| [mabughali/miia-pothole-train](https://huggingface.co/datasets/mabughali/miia-pothole-train) | vehicle | 4,026 dashcam images (South Africa) | Apache-2.0 |
| [aiswarya9302/RDD_2022India](https://huggingface.co/datasets/aiswarya9302/RDD_2022India) | vehicle | 7,706 images from RDD2022 India (Arya et al. 2022). Pothole = any D40 box; crack-only and undamaged photos count as normal. | CC BY-SA 4.0 |
| [RDD2022 Japan](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) | vehicle | 10,506 images, same labels. No split is given and neighbouring photos come from the same drive, so blocks of 100 consecutive photos go to train / val / test. | CC BY-SA 4.0 |
| [taroii/pothole-detection](https://huggingface.co/datasets/taroii/pothole-detection) | handheld | 437 images, pre-split | not stated |
| [Arpitraj01/Pothole_classification](https://huggingface.co/datasets/Arpitraj01/Pothole_classification) | handheld | ~490 images from 50 km of Kerala roads, severity labels → binary | MIT |
| [QR4Change](https://data.mendeley.com/datasets/zndzygc3p3) (test only) | handheld | 389 pothole photos taken on two phones in Pune | CC BY 4.0 |

Near-duplicate photos are removed with a perceptual hash (1,365 MIIA, 86 RDD India and 83 RDD Japan frames were
near-copies), so no photo sits in both train and test. Final split: 14,687 train / 3,470 val / 3,369 test, plus the
389 Pune photos.

Looked at and not used: the rest of QR4Change (its pothole photos are mostly web stock images and its no-pothole
photos come from other cameras, so a model would learn the camera); a "multi-weather" set whose night photos are
software-darkened day photos; the Nighttime Pothole Dataset (non-commercial licence); BharatPotHole, 7,000+ Indian
dashcam frames (CC BY-SA 4.0, needs a Kaggle login; a good next addition).

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
.venv/bin/python scripts/download_data.py         # ~3.5 GB into data/raw/ (add --clef for the 6.2 GB Clef-flash weights)
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train.py --model mobilenetv3_large_100 --size 384 --epochs 12 --name v3-mnv3-384
.venv/bin/python scripts/eval.py v3-384 --split val && .venv/bin/python scripts/eval.py v3-384
.venv/bin/python scripts/eval.py v3-384 --split field  # Pune street photos
.venv/bin/python scripts/tune_threshold.py v3-384 --recall-handheld 0.90 --recall-vehicle 0.85  # copy to models/v3-mnv3-384.thresholds.json
.venv/bin/python scripts/robustness.py v3-384
.venv/bin/python scripts/train_detector.py --epochs 14   # then save a half-precision copy as models/det-frcnn-mnv3.fp16.pth
.venv/bin/python scripts/eval_detector.py
.venv/bin/python scripts/eval.py detector --split val && .venv/bin/python scripts/eval.py detector
.venv/bin/python scripts/combine_vehicle.py --recall 0.85  # writes models/vehicle.json
.venv/bin/python scripts/eval.py v1 v2-384 siglip-zeroshot hf-resnet50
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

- **Night and low light** are the main weakness (vehicle AUC 0.73 when darkened and noisy). It needs real night-time training photos; the open "night" sets found so far were either non-commercial or software-darkened.
- **Small, distant potholes from Indian vehicles.** About 2 in 3 Indian potholes don't get a correctly placed box, and vehicle mode catches 66% of Indian potholes at the default cutoff (89% of Japanese ones). More Indian box-labelled data (BharatPotHole) is the next lever. Training longer (24 epochs) scored lower on validation (AP 0.450 vs 0.465).
- **The detector only knows Indian and Japanese vehicle footage.** Handheld mode doesn't use it.
- **Heavy video compression** costs accuracy (vehicle AUC 0.87 at JPEG quality 15). Upload frames at a reasonable quality.
- **Few handheld test photos** (89, of which 30 are pothole-free), so handheld numbers, especially false alarms, are less certain than vehicle ones. The Pune photos add catch-rate evidence but no false-alarm evidence.
- Off-the-shelf models are compared at their default cutoff; compare them on AUC.
- A YOLO pothole detector was left out: its weights ship as a pickle (`best.pt`), which can run code when loaded, and it is AGPL-licensed.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The datasets keep their own licences, listed under Data above. Models trained on RDD2022 data should credit it (CC BY-SA 4.0).
