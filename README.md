# Glacial Lake Semantic Segmentation

Binary segmentation of glacial lakes from Sentinel-2 satellite imagery using three deep learning architectures: U-Net, Simple CNN, and ASPP-SegNet.

---

## Project Structure

```
climate/
├── glacial_lake_segmentation/
│   ├── data/
│   │   ├── images/          # 410 Sentinel-2 tiles (400×400 PNG, Bands 8/4/3)
│   │   └── masks/           # 410 binary masks (same filenames as images)
│   ├── models/
│   │   ├── unet.py          # U-Net with skip connections
│   │   ├── simple_cnn.py    # Lightweight 3-block encoder-decoder
│   │   └── aspp_segnet.py   # SegNet encoder + ASPP module + decoder
│   ├── dataset/
│   │   └── glacial_lake_dataset.py  # Dataset, augmentation, data duplication
│   ├── utils/
│   │   ├── metrics.py       # IoU and F1 accumulator classes
│   │   └── visualization.py # Training curves and prediction grids
│   ├── api/
│   │   ├── main.py          # FastAPI application (serves React build + REST API)
│   │   └── schemas.py       # Pydantic response models
│   ├── config.py            # All hyperparameters and paths
│   ├── train.py             # Training script
│   ├── evaluate.py          # Validation evaluation script
│   ├── predict.py           # Single-image inference script
│   └── requirements.txt
├── README.md
└── project_context.md       # Precise architecture and training reference
```

---

## Requirements

- Python 3.9+
- CUDA 12.8 (RTX 5000 series)
- PyTorch nightly (CUDA 12.8 build)

---

## Installation

PyTorch nightly must be installed **before** the rest of the dependencies because the RTX 5060 Ti requires CUDA 12.8 which is not yet in a stable PyTorch release.

```powershell
cd "E:\Semester 6\CV\project\climate\glacial_lake_segmentation"

python -m venv venv
.\venv\Scripts\activate

# Step 1 — PyTorch nightly (CUDA 12.8, must come first)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 2 — everything else
pip install -r requirements.txt
```

---

## Data

The dataset is already present at `glacial_lake_segmentation/data/`:

- `data/images/` — 410 PNG tiles, 400×400 pixels, RGB composite of Sentinel-2 Bands 8 (NIR), 4 (Red), 3 (Green)
- `data/masks/` — 410 binary PNG masks, same filenames; lake pixels are white (255), background is black (0)

The dataset is split 80/20 (train/validation) deterministically at `RANDOM_SEED = 42`. Training images are duplicated once via deterministic horizontal flip before online augmentation is applied, effectively doubling the training set.

---

## Training

Run from inside `glacial_lake_segmentation/`:

```powershell
# Train one model
python train.py --model unet
python train.py --model simple_cnn
python train.py --model aspp_segnet

# Train all three sequentially
python train.py --model all
```

Optional overrides (defaults come from `config.py`):

```powershell
python train.py --model unet --epochs 25 --lr 0.001 --batch 8
```

**Outputs per model:**
- `checkpoints/{model}_epoch{N:02d}.pt` — checkpoint saved every epoch
- `checkpoints/{model}_best.pt` — overwritten whenever validation IoU improves
- `results/{model}_curves.png` — loss / IoU / F1 training curves

---

## Evaluation

Loads `{model}_best.pt` and runs the full validation split:

```powershell
# Evaluate one model
python evaluate.py --model unet

# Evaluate all three and write results/evaluation_results.csv
python evaluate.py --model all
```

Prints a table of Val IoU and Val F1. Saves a prediction grid to `results/{model}_predictions.png`.

---

## Single-Image Inference

```powershell
python predict.py --model unet --image path/to/tile.png
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `unet`, `simple_cnn`, or `aspp_segnet` |
| `--image` | required | Path to input PNG |
| `--output` | `results/pred_overlay.png` | Base path for output files |
| `--checkpoint` | `checkpoints/{model}_best.pt` | Override checkpoint path |

Produces three files from `--output`:
- `pred_overlay_mask.png` — binary mask (lake=255, background=0)
- `pred_overlay_colored_mask.png` — cyan lake on near-black background
- `pred_overlay.png` — original image with 55% cyan fill on lake pixels and red contour outlines

---

## API Server

The FastAPI backend serves both the REST API and the built React frontend.

```powershell
cd glacial_lake_segmentation
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` for the web UI.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Device, available models, loaded models |
| `POST` | `/predict` | Upload image + model name → base64 mask images + coverage % |
| `GET` | `/evaluate/{model_name}` | Run full validation set → IoU and F1 |

`POST /predict` accepts `multipart/form-data` with fields `image` (file) and `model_name` (string).

---

## Key Configuration (`config.py`)

| Parameter | Value |
|---|---|
| Image size | 400 × 400 |
| Batch size | 8 |
| Epochs | 25 |
| Learning rate | 1e-3 |
| Optimizer | Adam (weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=4) |
| Loss | BCEWithLogitsLoss |
| Threshold | 0.5 |
| Train/val split | 80 / 20 |
| Random seed | 42 |

---

## Reference

Lingling Xue et al., *Multi-Sensor Fusion and Deep Learning Approaches for Glacial Lake Segmentation*.
