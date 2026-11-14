# Test-set results

Held-out test split (pothole + normal), never seen in training. Models scored before RDD2022 Japan was added show the original 1,770-image test set in the Images column. Cutoff 0.5 for every model here; per-camera cutoffs for the default model are in README.

Sorted by AUC, which compares models without picking a cutoff (0.5 = coin flip, 1.0 = perfect).

| Model | Images | AUC | Accuracy | Precision | Recall | F1 | p50 latency | Device | Notes |
|---|---|---|---|---|---|---|---|---|---|
| v3-384 | 3,369 | 0.935 | 92.5% | 84.9% | 74.5% | 0.793 | 7.3 ms | cpu (onnxruntime) | Ours: v3-mnv3-384 fine-tuned on our train split |
| v2-384 | 3,369 | 0.834 | 84.6% | 60.4% | 57.5% | 0.589 | 11.7 ms | cpu (onnxruntime) | Ours: v2-mnv3-384 fine-tuned on our train split |
| v2-224 | 3,369 | 0.815 | 86.8% | 70.7% | 53.3% | 0.608 | 3.9 ms | cpu (onnxruntime) | Ours: v2-mnv3-224 fine-tuned on our train split |
| detector | 3,369 | 0.715 | 85.6% | 77.9% | 34.8% | 0.481 | 88.8 ms | cpu | Ours: Faster R-CNN MobileNetV3 detector, trained on RDD2022 India and Japan boxes; photo score = top box |
| v1-effnet | 3,369 | 0.677 | 56.8% | 26.2% | 68.9% | 0.380 | 6.6 ms | cpu (onnxruntime) | Ours: efficientnet_b0 fine-tuned on our train split |
| v1 | 3,369 | 0.668 | 51.7% | 25.5% | 79.1% | 0.386 | 3.6 ms | cpu (onnxruntime) | Ours: mobilenetv3_large_100 fine-tuned on our train split |
| clef-flash-zeroshot | 1,770 | 0.623 | 75.6% | 53.6% | 13.6% | 0.217 | 515.2 ms | mps (mlx, 4-bit) | Zero-shot Cloudflare clef-flash 9B (Qwen3.5-9B), 4-bit MLX, one yes/no question, no training |
| hf-resnet50 | 3,369 | 0.622 | 74.3% | 34.1% | 36.3% | 0.352 | 11.0 ms | mps | Off-the-shelf punchnami/resnet50-pothole-classification (training data unknown, may overlap our test set) |
| siglip-zeroshot | 3,369 | 0.601 | 80.4% | 46.2% | 13.3% | 0.206 | 32.3 ms | mps | Zero-shot google/siglip-base-patch16-224, prompt ensemble, no training |

## By camera type (AUC / accuracy)

| Model | handheld | vehicle |
|---|---|---|
| v3-384 | 0.880 / 77.5% | 0.943 / 93.0% |
| v2-384 | 0.871 / 79.8% | 0.837 / 84.7% |
| v2-224 | 0.854 / 77.5% | 0.813 / 87.0% |
| detector | 0.758 / 47.2% | 0.713 / 86.6% |
| v1-effnet | 0.812 / 75.3% | 0.674 / 56.2% |
| v1 | 0.828 / 76.4% | 0.668 / 51.0% |
| clef-flash-zeroshot | 0.884 / 73.0% | 0.591 / 75.7% |
| hf-resnet50 | 0.635 / 68.5% | 0.596 / 74.4% |
| siglip-zeroshot | 0.836 / 75.3% | 0.585 / 80.5% |

## By data source (AUC / accuracy)

| Model | kerala | miia | rdd_india | rdd_japan | taroii |
|---|---|---|---|---|---|
| v3-384 | 1.000 / 100.0% | 0.991 / 95.8% | 0.876 / 89.9% | 0.939 / 94.3% | 0.853 / 72.2% |
| v2-384 | 1.000 / 94.1% | 0.991 / 95.4% | 0.872 / 89.3% | 0.612 / 78.0% | 0.849 / 76.4% |
| v2-224 | 1.000 / 100.0% | 0.977 / 93.6% | 0.834 / 87.7% | 0.616 / 84.4% | 0.833 / 72.2% |
| detector | 0.760 / 35.3% | 0.546 / 55.2% | 0.861 / 89.1% | 0.943 / 95.0% | 0.751 / 50.0% |
| v1-effnet | 0.981 / 94.1% | 0.966 / 90.3% | 0.440 / 45.2% | 0.604 / 53.2% | 0.779 / 70.8% |
| v1 | 1.000 / 88.2% | 0.972 / 91.7% | 0.472 / 27.3% | 0.656 / 55.1% | 0.804 / 73.6% |
| clef-flash-zeroshot | 0.923 / 52.9% | 0.812 / 59.1% | 0.477 / 83.1% | - | 0.885 / 77.8% |
| hf-resnet50 | 0.154 / 76.5% | 0.917 / 64.9% | 0.549 / 75.4% | 0.718 / 76.8% | 0.675 / 66.7% |
| siglip-zeroshot | 0.885 / 76.5% | 0.671 / 55.2% | 0.420 / 87.2% | 0.650 / 83.9% | 0.821 / 75.0% |
