# Test-set results

Held-out test split (pothole + normal), never seen in training. Models scored before RDD2022 Japan was added show the original 1,770-image test set in the Images column. Cutoff 0.5 for every model here; per-camera cutoffs for the default model are in README.

Sorted by AUC, which compares models without picking a cutoff (0.5 = coin flip, 1.0 = perfect).

| Model | Images | AUC | Accuracy | Precision | Recall | F1 | p50 latency | Device | Notes |
|---|---|---|---|---|---|---|---|---|---|
| v4-384 | 4,008 | 0.909 | 89.1% | 74.7% | 71.1% | 0.729 | 7.3 ms | cpu (onnxruntime) | Ours: v4-mnv3-384 fine-tuned on our train split |
| v3-384 | 4,008 | 0.879 | 89.1% | 81.8% | 60.7% | 0.697 | 9.3 ms | cpu (onnxruntime) | Ours: v3-mnv3-384 fine-tuned on our train split |
| v2-384 | 4,008 | 0.805 | 82.2% | 58.4% | 47.6% | 0.525 | 10.1 ms | cpu (onnxruntime) | Ours: v2-mnv3-384 fine-tuned on our train split |
| v2-224 | 4,008 | 0.788 | 84.3% | 68.1% | 45.3% | 0.544 | 5.1 ms | cpu (onnxruntime) | Ours: v2-mnv3-224 fine-tuned on our train split |
| v1-effnet | 4,008 | 0.676 | 56.4% | 27.7% | 69.4% | 0.396 | 7.9 ms | cpu (onnxruntime) | Ours: efficientnet_b0 fine-tuned on our train split |
| v1 | 4,008 | 0.674 | 53.2% | 27.9% | 80.0% | 0.414 | 5.6 ms | cpu (onnxruntime) | Ours: mobilenetv3_large_100 fine-tuned on our train split |
| detector | 4,008 | 0.654 | 83.4% | 77.9% | 27.2% | 0.403 | 79.9 ms | cpu | Ours: Faster R-CNN MobileNetV3 detector, trained on RDD2022 India and Japan boxes; photo score = top box |
| hf-resnet50 | 4,008 | 0.638 | 73.7% | 36.7% | 37.8% | 0.372 | 9.8 ms | mps | Off-the-shelf punchnami/resnet50-pothole-classification (training data unknown, may overlap our test set) |
| clef-flash-zeroshot | 1,770 | 0.623 | 75.6% | 53.6% | 13.6% | 0.217 | 515.2 ms | mps (mlx, 4-bit) | Zero-shot Cloudflare clef-flash 9B (Qwen3.5-9B), 4-bit MLX, one yes/no question, no training |
| siglip-zeroshot | 4,008 | 0.618 | 79.2% | 48.0% | 11.5% | 0.185 | 37.7 ms | mps | Zero-shot google/siglip-base-patch16-224, prompt ensemble, no training |

## By camera type (AUC / accuracy)

| Model | handheld | vehicle |
|---|---|---|
| v4-384 | 0.857 / 78.7% | 0.917 / 89.3% |
| v3-384 | 0.880 / 77.5% | 0.882 / 89.4% |
| v2-384 | 0.871 / 79.8% | 0.805 / 82.2% |
| v2-224 | 0.854 / 77.5% | 0.785 / 84.5% |
| v1-effnet | 0.812 / 75.3% | 0.674 / 56.0% |
| v1 | 0.828 / 76.4% | 0.674 / 52.7% |
| detector | 0.758 / 47.2% | 0.647 / 84.2% |
| hf-resnet50 | 0.635 / 68.5% | 0.620 / 73.8% |
| clef-flash-zeroshot | 0.884 / 73.0% | 0.591 / 75.7% |
| siglip-zeroshot | 0.836 / 75.3% | 0.607 / 79.3% |

## By data source (AUC / accuracy)

| Model | bharat | kerala | miia | rdd_india | rdd_japan | taroii |
|---|---|---|---|---|---|---|
| v4-384 | 0.776 / 74.6% | 1.000 / 100.0% | 0.991 / 96.1% | 0.875 / 88.9% | 0.933 / 93.2% | 0.830 / 73.6% |
| v3-384 | 0.475 / 70.9% | 1.000 / 100.0% | 0.991 / 95.8% | 0.876 / 89.9% | 0.939 / 94.3% | 0.853 / 72.2% |
| v2-384 | 0.654 / 69.5% | 1.000 / 94.1% | 0.991 / 95.4% | 0.872 / 89.3% | 0.612 / 78.0% | 0.849 / 76.4% |
| v2-224 | 0.615 / 71.4% | 1.000 / 100.0% | 0.977 / 93.6% | 0.834 / 87.7% | 0.616 / 84.4% | 0.833 / 72.2% |
| v1-effnet | 0.672 / 54.5% | 0.981 / 94.1% | 0.966 / 90.3% | 0.440 / 45.2% | 0.604 / 53.2% | 0.779 / 70.8% |
| v1 | 0.738 / 61.0% | 1.000 / 88.2% | 0.972 / 91.7% | 0.472 / 27.3% | 0.656 / 55.1% | 0.804 / 73.6% |
| detector | 0.511 / 71.8% | 0.760 / 35.3% | 0.546 / 55.2% | 0.861 / 89.1% | 0.943 / 95.0% | 0.751 / 50.0% |
| hf-resnet50 | 0.704 / 70.6% | 0.154 / 76.5% | 0.917 / 64.9% | 0.549 / 75.4% | 0.718 / 76.8% | 0.675 / 66.7% |
| clef-flash-zeroshot | - | 0.923 / 52.9% | 0.812 / 59.1% | 0.477 / 83.1% | - | 0.885 / 77.8% |
| siglip-zeroshot | 0.706 / 72.8% | 0.885 / 76.5% | 0.671 / 55.2% | 0.420 / 87.2% | 0.650 / 83.9% | 0.821 / 75.0% |
