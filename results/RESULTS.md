# Test-set results

Held-out test split: 1770 images (pothole + normal), never seen in training. Cutoff 0.5 for every model here; per-camera cutoffs for the default model are in README.

Sorted by AUC, which compares models without picking a cutoff (0.5 = coin flip, 1.0 = perfect).

| Model | AUC | Accuracy | Precision | Recall | F1 | p50 latency | Device | Notes |
|---|---|---|---|---|---|---|---|---|
| v2-384 | 0.927 | 90.6% | 86.2% | 73.9% | 0.796 | 7.5 ms | cpu (onnxruntime) | Ours: v2-mnv3-384 fine-tuned on our train split |
| v2-224 | 0.908 | 88.9% | 81.1% | 72.3% | 0.764 | 3.9 ms | cpu (onnxruntime) | Ours: v2-mnv3-224 fine-tuned on our train split |
| v1-effnet | 0.715 | 59.9% | 35.0% | 71.6% | 0.471 | 7.2 ms | cpu (onnxruntime) | Ours: efficientnet_b0 fine-tuned on our train split |
| siglip-zeroshot | 0.665 | 77.2% | 97.4% | 8.6% | 0.159 | 32.1 ms | mps | Zero-shot google/siglip-base-patch16-224, prompt ensemble, no training |
| v1 | 0.647 | 48.6% | 30.4% | 82.5% | 0.444 | 4.9 ms | cpu (onnxruntime) | Ours: mobilenetv3_large_100 (v1, no Indian data) fine-tuned on our train split |
| hf-resnet50 | 0.644 | 72.0% | 41.0% | 29.1% | 0.340 | 14.1 ms | mps | Off-the-shelf punchnami/resnet50-pothole-classification (training data unknown, may overlap our test set) |
| clef-flash-zeroshot | 0.623 | 75.6% | 53.6% | 13.6% | 0.217 | 515.2 ms | mps (mlx, 4-bit) | Zero-shot Cloudflare clef-flash 9B (Qwen3.5-9B), 4-bit MLX, one yes/no question, no training |
| detector | 0.559 | 76.9% | 63.7% | 16.4% | 0.260 | 80.0 ms | cpu | Ours: Faster R-CNN MobileNetV3 detector, trained on RDD2022 India boxes; photo score = top box |

## By camera type (AUC / accuracy)

| Model | handheld | vehicle |
|---|---|---|
| v2-384 | 0.871 / 79.8% | 0.943 / 91.1% |
| v2-224 | 0.854 / 77.5% | 0.921 / 89.5% |
| v1-effnet | 0.812 / 75.3% | 0.716 / 59.1% |
| siglip-zeroshot | 0.836 / 75.3% | 0.650 / 77.3% |
| v1 | 0.828 / 76.4% | 0.648 / 47.2% |
| hf-resnet50 | 0.635 / 68.5% | 0.609 / 72.2% |
| clef-flash-zeroshot | 0.884 / 73.0% | 0.591 / 75.7% |
| detector | 0.564 / 34.8% | 0.572 / 79.1% |

## By data source (AUC / accuracy)

| Model | kerala | miia | rdd_india | taroii |
|---|---|---|---|---|
| v2-384 | 1.000 / 94.1% | 0.991 / 95.4% | 0.872 / 89.3% | 0.849 / 76.4% |
| v2-224 | 1.000 / 100.0% | 0.977 / 93.6% | 0.834 / 87.7% | 0.833 / 72.2% |
| v1-effnet | 0.981 / 94.1% | 0.966 / 90.3% | 0.440 / 45.2% | 0.779 / 70.8% |
| siglip-zeroshot | 0.885 / 76.5% | 0.671 / 55.2% | 0.420 / 87.2% | 0.821 / 75.0% |
| v1 | 1.000 / 88.2% | 0.972 / 91.7% | 0.472 / 27.3% | 0.804 / 73.6% |
| hf-resnet50 | 0.154 / 76.5% | 0.917 / 64.9% | 0.549 / 75.4% | 0.675 / 66.7% |
| clef-flash-zeroshot | 0.923 / 52.9% | 0.812 / 59.1% | 0.477 / 83.1% | 0.885 / 77.8% |
| detector | 0.500 / 23.5% | 0.502 / 55.2% | 0.859 / 89.8% | 0.589 / 37.5% |
