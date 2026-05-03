import torch
from pathlib import Path

# --- Paths ---
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
IMAGE_DIR       = DATA_DIR / "images"
MASK_DIR        = DATA_DIR / "masks"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
RESULTS_DIR     = BASE_DIR / "results"

# --- Training Hyperparameters (from paper Table I) ---
IMAGE_SIZE    = (256, 256)
BATCH_SIZE    = 16
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-3
NUM_WORKERS   = 4
PIN_MEMORY    = True
RANDOM_SEED   = 42
TRAIN_SPLIT   = 0.8
VAL_SPLIT     = 0.2

# --- Regularization (paper-silent; light L2 weight decay for Adam) ---
WEIGHT_DECAY  = 1e-5

# --- Inference ---
THRESHOLD = 0.5   # sigmoid threshold for binary mask

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model registry keys ---
MODELS = ["unet", "simple_cnn", "aspp_segnet"]
