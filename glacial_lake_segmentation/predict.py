import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

import config
from models import get_model


def load_model_from_checkpoint(model_name: str, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    model = get_model(model_name)
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess_image_file(image_path) -> torch.Tensor:
    """
    Paper spec: 400×400 input tiles, normalized to [0, 1].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # normalize to [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(config.DEVICE)


def run_inference(model, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        binary_np  : (H, W) uint8 array of {0, 1}  — the true binary mask
        prob_np    : (H, W) float32 array of [0, 1] — raw sigmoid probabilities
    Model outputs raw logits; sigmoid converts to probabilities;
    threshold at config.THRESHOLD (0.5) produces the binary mask.
    """
    with torch.no_grad():
        logits   = model(tensor)                    # raw logits (unbounded)
        prob_np  = torch.sigmoid(logits).squeeze().cpu().numpy()   # [0, 1]
    binary_np = (prob_np >= config.THRESHOLD).astype(np.uint8)    # {0, 1}
    return binary_np, prob_np


def main():
    parser = argparse.ArgumentParser(description="Single-image glacial lake prediction")
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--image",      type=str, required=True)
    parser.add_argument("--output",     type=str, default=str(config.RESULTS_DIR / "pred_overlay.png"))
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    model      = load_model_from_checkpoint(args.model, args.checkpoint)
    tensor     = preprocess_image_file(args.image)
    binary_np, prob_np = run_inference(model, tensor)

    total_pixels = binary_np.size
    lake_pixels  = int(binary_np.sum())

    output_path  = Path(args.output)
    mask_path    = output_path.parent / (output_path.stem + "_mask.png")
    colored_path = output_path.parent / (output_path.stem + "_colored_mask.png")

    # ── 1. Save pure binary mask (lake=255, background=0) ────────────────────
    mask_img = Image.fromarray((binary_np * 255).astype(np.uint8), mode="L")
    mask_img.save(mask_path)

    # ── 2. Save colored mask (lake=cyan, background=near-black) ──────────────
    h, w = binary_np.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[binary_np == 0] = [10, 10, 20]      # near-black background
    rgb_mask[binary_np == 1] = [0, 220, 255]     # cyan lake
    Image.fromarray(rgb_mask, mode="RGB").save(colored_path)

    # ── 3. Save overlay (original + semi-transparent fill + red contour) ──────
    orig = np.array(
        Image.open(args.image).convert("RGB").resize(
            (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
        ),
        dtype=np.uint8,
    )
    overlay = orig.copy()
    lake_region = binary_np.astype(bool)
    cyan = np.array([0, 220, 255], dtype=np.uint8)
    overlay[lake_region] = (overlay[lake_region] * 0.45 + cyan * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
    Image.fromarray(overlay).save(output_path)

    print(f"Model        : {args.model}")
    print(f"Image        : {args.image}")
    print(f"Input size   : {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]} (paper spec)")
    print(f"Lake pixels  : {lake_pixels} / {total_pixels} ({100*lake_pixels/total_pixels:.2f}%)")
    print(f"Binary mask  : {mask_path}")
    print(f"Colored mask : {colored_path}")
    print(f"Overlay      : {output_path}")


if __name__ == "__main__":
    main()
