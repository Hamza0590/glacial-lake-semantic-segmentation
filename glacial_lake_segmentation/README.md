# Glacial Lake Segmentation

## Overview

This project reproduces the benchmark results from the IEEE paper *"Multi-Sensor Fusion and Deep Learning Approaches for Semantic Segmentation of Glacial Lakes"*. It trains and evaluates three deep learning architectures — U-Net, Simple CNN, and ASPP SegNet — on Sentinel-2 satellite imagery to perform binary semantic segmentation of glacial lakes. A FastAPI backend with a vanilla-JS frontend is included for interactive single-image inference.

## Folder Structure

```
glacial_lake_segmentation/
├── config.py                  — All paths and hyperparameters (single source of truth)
├── train.py                   — Training script for all three models
├── evaluate.py                — Evaluation script; outputs metrics and prediction grids
├── predict.py                 — Single-image CLI inference with overlay output
├── requirements.txt           — Python dependencies (see install order below)
├── README.md                  — This file
├── data/
│   ├── images/                — Input Sentinel-2 PNG patches (place Kaggle data here)
│   └── masks/                 — Binary ground-truth mask PNGs (place Kaggle data here)
├── dataset/
│   ├── __init__.py            — Package init; exports GlacialLakeDataset, get_dataloaders
│   └── glacial_lake_dataset.py — Dataset class and dataloader factory with augmentation
├── models/
│   ├── __init__.py            — Model factory (get_model) and parameter counter
│   ├── unet.py                — U-Net with encoder/decoder skip connections
│   ├── simple_cnn.py          — Lightweight CNN baseline (no skip connections)
│   └── aspp_segnet.py         — SegNet encoder + ASPP multi-scale context module
├── utils/
│   ├── __init__.py            — Package init
│   ├── metrics.py             — Stateful IoU/F1 accumulators and batch metric helper
│   └── visualization.py       — Prediction grid and training curve plotting utilities
├── checkpoints/               — Saved model weights (.pt files written during training)
│   └── .gitkeep
├── results/                   — Output PNGs and CSVs (curves, grids, evaluation results)
│   └── .gitkeep
└── api/
    ├── __init__.py            — Package init
    ├── main.py                — FastAPI application with /predict, /evaluate, /health
    ├── schemas.py             — Pydantic request/response models
    └── static/
        └── index.html         — Self-contained frontend (plain HTML/CSS/JS, no build step)
```

## Installation

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd glacial_lake_segmentation
   ```

2. **Install PyTorch nightly (RTX 5 series — CUDA 12.8, must be done first)**
   ```bash
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. Download the dataset from Kaggle:
   [https://www.kaggle.com/datasets/aatishshresthaa/glacial-lake-dataset](https://www.kaggle.com/datasets/aatishshresthaa/glacial-lake-dataset)

2. Extract and place files:
   - Image patches (PNG) → `data/images/`
   - Binary mask PNGs  → `data/masks/`

   Each mask filename must match its corresponding image filename exactly (e.g., `patch_001.png` ↔ `patch_001.png`).

## Usage

### Training

```bash
python train.py --model unet
python train.py --model simple_cnn
python train.py --model aspp_segnet
python train.py --model all        # trains all three sequentially
```

Optional flags: `--epochs`, `--lr`, `--batch`

### Evaluation

```bash
python evaluate.py --model unet
python evaluate.py --model all     # evaluates all three; saves evaluation_results.csv
```

### Single Image Prediction

```bash
python predict.py --model unet --image path/to/image.png
python predict.py --model unet --image path/to/image.png --output path/to/save.png
```

### Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000)

## Expected Results (from paper)

| Model       | Val F1  | Val IoU | Val Loss |
|-------------|---------|---------|----------|
| U-Net       | 0.9438  | 0.8961  | 0.03898  |
| Simple CNN  | 0.9557  | 0.9155  | 0.0361   |
| ASPP SegNet | 0.9542  | 0.9129  | 0.03337  |
