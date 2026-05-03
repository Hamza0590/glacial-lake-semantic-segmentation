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
    img = Image.open(image_path).convert("RGB")
    img = img.resize((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(config.DEVICE)


def run_inference(model, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        preds = torch.sigmoid(model(tensor))                # model outputs logits; sigmoid here
    binary = (preds.squeeze().cpu().numpy() >= config.THRESHOLD).astype(np.uint8)
    return binary


def main():
    parser = argparse.ArgumentParser(description="Single-image glacial lake prediction")
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--image",      type=str, required=True)
    parser.add_argument("--output",     type=str, default=str(config.RESULTS_DIR / "pred_overlay.png"))
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.model, args.checkpoint)
    tensor = preprocess_image_file(args.image)
    binary_mask = run_inference(model, tensor)

    total_pixels = binary_mask.size
    lake_pixels  = int(binary_mask.sum())

    output_path = Path(args.output)
    mask_path   = output_path.parent / (output_path.stem + "_mask.png")

    # Save binary mask (0 or 255)
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    mask_img.save(mask_path)

    # Save overlay with red contours
    orig = np.array(Image.open(args.image).convert("RGB").resize(
        (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), Image.BILINEAR
    ))
    overlay = orig.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    Image.fromarray(overlay).save(output_path)

    print(f"Model      : {args.model}")
    print(f"Image      : {args.image}")
    print(f"Lake pixels: {lake_pixels} / {total_pixels} ({100*lake_pixels/total_pixels:.2f}%)")
    print(f"Mask saved : {mask_path}")
    print(f"Overlay    : {output_path}")


if __name__ == "__main__":
    main()
