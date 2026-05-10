# Project Context: Glacial Lake Semantic Segmentation

Precise technical reference derived directly from the implementation. All values are ground-truth from source code, not from the paper.

---

## 1. Problem

Binary pixel-level segmentation of glacial lakes from Sentinel-2 satellite tiles. Each pixel is classified as lake (1) or background (0). The model must handle mountain shadows, turbid water, and scale variation from small supraglacial ponds to large proglacial lakes.

---

## 2. Dataset

| Property | Value |
|---|---|
| Total samples | 410 image / mask pairs |
| Image format | PNG, RGB, 400×400 pixels |
| Bands | Sentinel-2 B8 (NIR), B4 (Red), B3 (Green) |
| Mask format | PNG, grayscale; lake=255, background=0 |
| Train split | 328 images (80%) |
| Validation split | 82 images (20%) |
| Split seed | 42 (sklearn `train_test_split`, shuffle=True) |
| Effective train size | 656 samples (2× via data duplication) |

---

## 3. Data Pipeline

### Normalization
Images are divided by 255.0 at load time (`__getitem__`), bringing pixel values into [0, 1]. Albumentations `Normalize(mean=0, std=1)` is a no-op that preserves this range.

Masks are binarized: `(mask / 255.0 > 0).astype(float32)` → {0.0, 1.0}.

### Data Duplication
`GlacialLakeDataset(duplicate=True)` doubles the training set by exposing each sample twice. Samples at index `i >= n_original` return the horizontally-flipped version of sample `i - n_original`, applied deterministically with `np.fliplr` before any online transforms run.

This is applied only to the training set. The validation set uses `duplicate=False`.

### Online Augmentation (training only)

| Transform | Parameters |
|---|---|
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.5 |
| Rotate | limit=±15°, p=0.5 |
| RandomScale | scale_limit=±10%, p=0.5 |
| Resize | 400×400 (applied last, normalizes RandomScale output) |

Validation uses only Resize to 400×400 — no augmentation.

### DataLoader

| Parameter | Value |
|---|---|
| Batch size | 8 |
| Shuffle | True (train), False (val) |
| num_workers | 4 |
| pin_memory | True |

Tensors are `.clone()`'d inside `__getitem__` to ensure owned storage — required on Windows with `num_workers > 0` because `ToTensorV2` produces tensors backed by numpy memory.

---

## 4. Model Architectures

All models:
- Accept input shape `(B, 3, 400, 400)`, float32, values in [0, 1]
- Output raw logits of shape `(B, 1, 400, 400)` — **no sigmoid inside the model**
- Use `kaiming_normal_` weight initialization (`fan_out`, `nonlinearity='relu'`)

### 4.1 U-Net (`models/unet.py`)

**Encoder** — 4 blocks, each: Conv2d(3×3, pad=1) → BN → ReLU → Conv2d(3×3, pad=1) → BN → ReLU, followed by MaxPool2d(2×2).

| Block | Input ch | Output ch |
|---|---|---|
| enc1 | 3 | 64 |
| enc2 | 64 | 128 |
| enc3 | 128 | 256 |
| enc4 | 256 | 512 |

Spatial progression: 400 → 200 → 100 → 50 → 25

**Bottleneck** — same conv block structure: 512 → 1024 channels at 25×25.

**Decoder** — 4 stages. Each: ConvTranspose2d(2×2, stride=2) upsample → concat with corresponding encoder skip → conv block.

| Stage | Upsample in→out ch | Skip concat | Conv block ch |
|---|---|---|---|
| dec4 | 1024→512 | +enc4 (512) | 1024→512 |
| dec3 | 512→256 | +enc3 (256) | 512→256 |
| dec2 | 256→128 | +enc2 (128) | 256→128 |
| dec1 | 128→64 | +enc1 (64) | 128→64 |

Spatial progression: 25 → 50 → 100 → 200 → 400

**Head** — Conv2d(64→1, kernel=1×1), raw logits.

Skip connections use `F.interpolate` (bilinear) to align spatial dims if there is a 1-pixel mismatch from odd input sizes.

---

### 4.2 Simple CNN (`models/simple_cnn.py`)

**Encoder** (sequential):

| Layer | Config | Output shape |
|---|---|---|
| Conv2d + ReLU | 3→32, 3×3, pad=1 | (32, 400, 400) |
| MaxPool2d | 2×2 | (32, 200, 200) |
| Conv2d + ReLU | 32→64, 3×3, pad=1 | (64, 200, 200) |
| MaxPool2d | 2×2 | (64, 100, 100) |
| Conv2d + ReLU | 64→128, 3×3, pad=1 | (128, 100, 100) |
| MaxPool2d | 2×2 | (128, 50, 50) |

**Decoder** (sequential):

| Layer | Config | Output shape |
|---|---|---|
| Upsample | scale=2, bilinear | (128, 100, 100) |
| Conv2d + ReLU | 128→64, 3×3, pad=1 | (64, 100, 100) |
| Upsample | scale=2, bilinear | (64, 200, 200) |
| Conv2d + ReLU | 64→32, 3×3, pad=1 | (32, 200, 200) |
| Upsample | scale=2, bilinear | (32, 400, 400) |

**Head** — Conv2d(32→1, 1×1), raw logits. Output shape: (1, 400, 400).

No skip connections. No BatchNorm. All convolutions use `padding=1` ("same" padding for 3×3 kernels).

---

### 4.3 ASPP-SegNet (`models/aspp_segnet.py`)

**Encoder** — 3 blocks, each: Conv2d(3×3, pad=1) → BN → ReLU → Conv2d(3×3, pad=1) → BN → ReLU → MaxPool2d(2×2).

| Block | Input ch | Output ch | Spatial out |
|---|---|---|---|
| enc1 | 3 | 64 | 200×200 |
| enc2 | 64 | 128 | 100×100 |
| enc3 | 128 | 256 | 50×50 |

**ASPP Module** — receives 256 channels at 50×50. Five parallel branches:

| Branch | Operation | Output ch |
|---|---|---|
| b1 | Conv2d(1×1) → BN → ReLU | 256 |
| b2 | Conv2d(3×3, dilation=6, pad=6) → BN → ReLU | 256 |
| b3 | Conv2d(3×3, dilation=12, pad=12) → BN → ReLU | 256 |
| b4 | Conv2d(3×3, dilation=18, pad=18) → BN → ReLU | 256 |
| b5 | AdaptiveAvgPool2d(1) → Conv2d(1×1) → BN → ReLU → bilinear upsample to 50×50 | 256 |

All 5 branches concatenated → 1280 channels → fusion Conv2d(1×1, 1280→256) → BN → ReLU → 256 channels at 50×50.

**Decoder** — 3 stages, each: Upsample(scale=2, bilinear) → Conv2d(3×3, pad=1) → BN → ReLU.

| Stage | Input ch | Output ch | Spatial out |
|---|---|---|---|
| dec3 | 256 | 128 | 100×100 |
| dec2 | 128 | 64 | 200×200 |
| dec1 | 64 | 64 | 400×400 |

No skip connections from encoder.

**Head** — Conv2d(64→1, 1×1), raw logits.

---

## 5. Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Optimizer | Adam | `torch.optim.Adam` |
| Learning rate | 1e-3 | `config.LEARNING_RATE` |
| Weight decay | 1e-5 | `config.WEIGHT_DECAY` |
| Loss | BCEWithLogitsLoss | numerically stable; sigmoid + BCE fused |
| Epochs | 25 | `config.NUM_EPOCHS` |
| Batch size | 8 | `config.BATCH_SIZE` |
| LR scheduler | ReduceLROnPlateau | mode=max (val IoU), factor=0.5, patience=4, min_lr=1e-6 |
| Gradient clipping | max_norm=1.0 | `torch.nn.utils.clip_grad_norm_` |
| Device | CUDA if available | RTX 5060 Ti / CUDA 12.8 / PyTorch nightly |

Checkpoints are saved every epoch. The best checkpoint (highest val IoU) is separately saved as `{model}_best.pt`.

A sanity check at the start of every training run asserts that model outputs contain values outside [0, 1], confirming sigmoid is not inside the model head.

---

## 6. Metrics

Metrics are computed with custom accumulator classes (`IoUMetric`, `F1Metric`) that sum TP, FP, FN across all batches before computing the final value. The threshold is 0.5.

```
IoU  = TP / (TP + FP + FN + 1e-8)
F1   = 2·TP / (2·TP + FP + FN + 1e-8)
```

Both metrics are computed per epoch for both training and validation sets. The LR scheduler steps on val IoU.

---

## 7. Inference Pipeline

1. Load image as RGB, resize to 400×400 (bilinear), divide by 255.0.
2. Convert to tensor `(1, 3, 400, 400)`, move to device.
3. Forward pass → raw logits `(1, 1, 400, 400)`.
4. `torch.sigmoid(logits)` → probabilities in [0, 1].
5. `>= 0.5` threshold → binary mask `(400, 400)` with values {0, 1}.

### Output Formats

| Output | Description |
|---|---|
| Binary mask | Grayscale PNG; lake=255, background=0 |
| Colored mask | RGB PNG; lake=cyan (0, 220, 255), background=near-black (10, 10, 20) |
| Overlay | Original image with 55% cyan blend on lake pixels + red (255, 50, 50) contour outlines via `cv2.findContours` |

---

## 8. API (`api/main.py`)

FastAPI application. Loads model checkpoints on first request and caches them in `_model_cache`.

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | — | device, available models, loaded models |
| POST | `/predict` | `multipart/form-data`: `image` file + `model_name` string | `PredictionResponse` |
| GET | `/evaluate/{model_name}` | — | `EvaluationResponse` |
| GET | `/{full_path}` | — | Serves React build (`api/static/`) |

**`PredictionResponse` fields:**
- `model_name: str`
- `lake_coverage_percent: float`
- `mask_image_base64: str`
- `colored_mask_base64: str`
- `overlay_image_base64: str`

**`EvaluationResponse` fields:**
- `model_name: str`
- `val_iou: float`
- `val_f1: float`
- `predictions_image_path: str`

Run: `uvicorn api.main:app --reload --port 8000` from inside `glacial_lake_segmentation/`.

---

## 9. PyTorch vs. Paper (TensorFlow/Keras)

The reference paper specifies a TensorFlow/Keras implementation. This codebase is PyTorch. Key adaptations:

| Paper spec | This implementation | Reason |
|---|---|---|
| Sigmoid in model head | Sigmoid removed from model; applied at inference | `BCEWithLogitsLoss` is numerically stable (log-sum-exp trick avoids log(0) for saturated outputs) |
| Custom Keras Metric API | `IoUMetric` / `F1Metric` Python classes | PyTorch has no built-in Metric API |
| No BatchNorm in U-Net | BatchNorm2d added to U-Net conv blocks | Training stability; standard PyTorch U-Net practice |
| No LR scheduler | ReduceLROnPlateau on val IoU | Prevents stagnation; does not contradict paper's lr=1e-3 start |
