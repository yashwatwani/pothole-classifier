# Test-set results

Held-out test split: 607 images (pothole + normal), never seen in training. Threshold 0.5.

| Model | Accuracy | Precision | Recall | F1 | AUC | p50 latency | Device | Notes |
|---|---|---|---|---|---|---|---|---|
| ours | 89.5% | 92.8% | 84.5% | 0.885 | 0.949 | 3.4 ms | cpu (onnxruntime) | Ours: mobilenetv3_large_100 fine-tuned on our train split |
| ours-effnet | 88.1% | 89.5% | 85.2% | 0.873 | 0.934 | 5.6 ms | cpu (onnxruntime) | Ours: efficientnet_b0 fine-tuned on our train split |
| hf-resnet50 | 65.4% | 82.4% | 35.4% | 0.495 | 0.872 | 13.0 ms | mps | Off-the-shelf punchnami/resnet50-pothole-classification (training data unknown, may overlap our test set) |
| siglip-zeroshot | 58.2% | 97.4% | 13.1% | 0.230 | 0.695 | 31.5 ms | mps | Zero-shot google/siglip-base-patch16-224, prompt ensemble, no training |

## F1 by data source

| Model | kerala | miia | taroii |
|---|---|---|---|
| ours | 0.929 (n=17) | 0.904 (n=518) | 0.765 (n=72) |
| ours-effnet | 0.963 (n=17) | 0.892 (n=518) | 0.734 (n=72) |
| hf-resnet50 | 0.867 (n=17) | 0.355 (n=518) | 0.769 (n=72) |
| siglip-zeroshot | 0.833 (n=17) | 0.000 (n=518) | 0.757 (n=72) |
