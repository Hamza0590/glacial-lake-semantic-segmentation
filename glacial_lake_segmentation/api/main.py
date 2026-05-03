import io
import base64
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models import get_model
from api.schemas import PredictionResponse, HealthResponse, EvaluationResponse
from utils.metrics import IoUMetric, F1Metric
from utils.visualization import save_prediction_grid
from dataset import get_dataloaders

app = FastAPI(title="Glacial Lake Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_model_cache: dict[str, torch.nn.Module] = {}


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> torch.nn.Module:
    if model_name in _model_cache:
        return _model_cache[model_name]
    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint for '{model_name}' not found. "
                   f"Train first: python train.py --model {model_name}",
        )
    model = get_model(model_name)
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _model_cache[model_name] = model
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
    )
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(config.DEVICE)


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    arr = tensor.squeeze().cpu().numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def rgb_array_to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    available = [m for m in config.MODELS
                 if (config.CHECKPOINT_DIR / f"{m}_best.pt").exists()]
    return HealthResponse(
        status="ok",
        device=config.DEVICE,
        available_models=available,
        loaded_models=list(_model_cache.keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile, model_name: str = Form(...)):
    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint for '{model_name}' not found. "
                   f"Train first: python train.py --model {model_name}",
        )

    model = load_model(model_name)

    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    tensor = preprocess_image(pil_image)

    with torch.no_grad():
        preds = torch.sigmoid(model(tensor))                # model outputs logits; sigmoid here

    binary = (preds >= config.THRESHOLD).float()
    binary_np = binary.squeeze().cpu().numpy().astype(np.uint8)

    total_pixels = binary_np.size
    lake_pixels  = int(binary_np.sum())
    coverage     = round(100.0 * lake_pixels / total_pixels, 4)

    # Mask PNG (in memory)
    mask_b64 = tensor_to_base64_png(binary)

    # Overlay PNG (in memory)
    orig = np.array(
        pil_image.convert("RGB").resize(
            (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
        )
    )
    overlay = orig.copy()
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    overlay_b64 = rgb_array_to_base64_png(overlay)

    return PredictionResponse(
        model_name=model_name,
        lake_coverage_percent=coverage,
        mask_image_base64=mask_b64,
        overlay_image_base64=overlay_b64,
    )


@app.get("/evaluate/{model_name}", response_model=EvaluationResponse)
def evaluate(model_name: str):
    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint for '{model_name}' not found. "
                   f"Train first: python train.py --model {model_name}",
        )

    model = load_model(model_name)

    _, val_loader = get_dataloaders(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        train_split=config.TRAIN_SPLIT,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        seed=config.RANDOM_SEED,
    )

    iou_acc = IoUMetric()
    f1_acc  = F1Metric()
    sample_imgs, sample_masks_list, sample_preds_list = [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(config.DEVICE)
            masks  = masks.to(config.DEVICE)
            preds  = torch.sigmoid(model(images))           # model outputs logits; sigmoid here
            iou_acc.update(preds, masks)
            f1_acc.update(preds, masks)
            if len(sample_imgs) < 8:
                n = min(8 - len(sample_imgs), images.size(0))
                sample_imgs.extend(images[:n].cpu())
                sample_masks_list.extend(masks[:n].cpu())
                binary = (preds >= config.THRESHOLD).float()
                sample_preds_list.extend(binary[:n].cpu())

    grid_path = config.RESULTS_DIR / f"{model_name}_predictions.png"
    save_prediction_grid(sample_imgs, sample_masks_list, sample_preds_list, grid_path, n_samples=8)

    return EvaluationResponse(
        model_name=model_name,
        val_iou=round(iou_acc.compute(), 6),
        val_f1=round(f1_acc.compute(), 6),
        predictions_image_path=str(grid_path.relative_to(config.BASE_DIR)),
    )
