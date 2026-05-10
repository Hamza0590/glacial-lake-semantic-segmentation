from pydantic import BaseModel

from typing import Optional

class PredictionResponse(BaseModel):
    model_name: str
    lake_coverage_percent: float
    mask_image_base64: str        # base64-encoded grayscale PNG of binary mask (0/255)
    colored_mask_base64: str      # base64-encoded RGB PNG: lake=cyan, background=black
    overlay_image_base64: str     # base64-encoded RGB PNG with red contour overlay
    feature_maps: Optional[dict[str, str]] = None  # Mapping of layer ID to base64 image
    inference_time_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str                   # "ok"
    device: str                   # e.g. "cuda" or "cpu"
    available_models: list[str]   # models whose best checkpoint exists on disk
    loaded_models: list[str]      # models currently cached in memory


class EvaluationResponse(BaseModel):
    model_name: str
    val_iou: float
    val_f1: float
    predictions_image_path: str   # relative path to saved prediction grid PNG
