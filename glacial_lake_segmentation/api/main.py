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

# Mount the entire static directory to serve assets like /assets/index.js
app.mount("/assets", StaticFiles(directory=str(Path(__file__).parent / "static" / "assets")), name="assets")

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
    """
    Paper spec: 400×400 input tiles from Sentinel-2 (Bands 8, 4, 3),
    normalized to [0, 1].  We accept any RGB upload and resize to match.
    """
    image = image.convert("RGB").resize(
        (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
    )
    arr = np.array(image, dtype=np.float32) / 255.0   # normalize to [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.to(config.DEVICE)


def binary_mask_to_base64(binary_np: np.ndarray) -> str:
    """Return a grayscale PNG where lake pixels = 255, background = 0."""
    mask_uint8 = (binary_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def colored_mask_to_base64(binary_np: np.ndarray) -> str:
    """
    Return a color-mapped binary mask for visualization:
      lake pixels    → bright cyan  (0, 220, 255)
      background     → near-black   (10,  10,  20)
    This matches the false-colour NIR composite feel of the paper.
    """
    h, w = binary_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[binary_np == 0] = [10, 10, 20]       # background: near-black
    rgb[binary_np == 1] = [0, 220, 255]      # lake: cyan
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def overlay_to_base64(orig_rgb: np.ndarray, binary_np: np.ndarray) -> str:
    """
    Draw red contours on the original image to show lake boundaries.
    Uses a semi-transparent cyan fill for detected lake area + red contour.
    """
    overlay = orig_rgb.copy().astype(np.uint8)

    # Semi-transparent cyan fill for lake region
    lake_region = binary_np.astype(bool)
    cyan = np.array([0, 220, 255], dtype=np.uint8)
    overlay[lake_region] = (overlay[lake_region] * 0.45 + cyan * 0.55).astype(np.uint8)

    # Red contour outline
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)

    img = Image.fromarray(overlay, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def rgb_array_to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Routes ─────────────────────────────────────────────────────────────────

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

    # ── Preprocessing (paper: 400×400, normalized to [0,1]) ──────────────────
    tensor = preprocess_image(pil_image)

    # ── Inference ─────────────────────────────────────────────────────────────
    # Models output raw logits; sigmoid converts to probabilities [0, 1].
    # Threshold at 0.5 → binary mask {0, 1} as per paper.
    with torch.no_grad():
        logits = model(tensor)                          # (1, 1, H, W) raw logits
        probs  = torch.sigmoid(logits)                  # probabilities [0, 1]

    binary    = (probs >= config.THRESHOLD).float()     # {0.0, 1.0}
    binary_np = binary.squeeze().cpu().numpy().astype(np.uint8)   # (H, W) {0, 1}

    total_pixels = binary_np.size
    lake_pixels  = int(binary_np.sum())
    coverage     = round(100.0 * lake_pixels / total_pixels, 4)

    # ── Resize original for overlays ─────────────────────────────────────────
    orig_resized = np.array(
        pil_image.convert("RGB").resize(
            (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
        ),
        dtype=np.uint8,
    )

    # ── Build response images ─────────────────────────────────────────────────
    mask_b64         = binary_mask_to_base64(binary_np)          # pure B&W mask
    colored_mask_b64 = colored_mask_to_base64(binary_np)         # cyan-on-black
    overlay_b64      = overlay_to_base64(orig_resized, binary_np) # orig + cyan fill + red contour

    return PredictionResponse(
        model_name=model_name,
        lake_coverage_percent=coverage,
        mask_image_base64=mask_b64,
        colored_mask_base64=colored_mask_b64,
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
            preds  = torch.sigmoid(model(images))       # probabilities
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


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    static_file = Path(__file__).parent / "static" / full_path
    if static_file.is_file():
        return FileResponse(str(static_file))
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))
