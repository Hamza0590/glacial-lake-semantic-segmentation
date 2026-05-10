import argparse
import random
import csv
import torch

import config
from dataset import get_dataloaders
from models import get_model
from utils.metrics import IoUMetric, F1Metric
from utils.visualization import save_prediction_grid


def evaluate_model(model_name: str, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"

    model = get_model(model_name)
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    sample_images, sample_masks, sample_preds = [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images  = images.to(config.DEVICE)
            masks   = masks.to(config.DEVICE)
            preds   = model(images)                         # model outputs probabilities
            iou_acc.update(preds, masks)
            f1_acc.update(preds, masks)

            if len(sample_images) < 8:
                n = min(8 - len(sample_images), images.size(0))
                sample_images.extend(images[:n].cpu())
                sample_masks.extend(masks[:n].cpu())
                binary = (preds >= config.THRESHOLD).float()
                sample_preds.extend(binary[:n].cpu())

    val_iou = iou_acc.compute()
    val_f1  = f1_acc.compute()

    print("┌─────────────────┬──────────┬──────────┐")
    print("│ Model           │ Val IoU  │ Val F1   │")
    print("├─────────────────┼──────────┼──────────┤")
    print(f"│ {model_name:<15} │ {val_iou:.4f}   │ {val_f1:.4f}   │")
    print("└─────────────────┴──────────┴──────────┘")

    grid_path = config.RESULTS_DIR / f"{model_name}_predictions.png"
    save_prediction_grid(sample_images, sample_masks, sample_preds, grid_path, n_samples=8)
    print(f"Prediction grid saved to: {grid_path}")

    return val_iou, val_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate glacial lake segmentation models")
    parser.add_argument("--model",      type=str, required=True,
                        help="Model name or 'all'. Choices: unet, simple_cnn, aspp_segnet, all")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.model == "all":
        results = []
        for model_name in config.MODELS:
            iou, f1 = evaluate_model(model_name)
            results.append({"model_name": model_name, "val_iou": iou, "val_f1": f1})

        print("\n=== Combined Results ===")
        print("┌─────────────────┬──────────┬──────────┐")
        print("│ Model           │ Val IoU  │ Val F1   │")
        print("├─────────────────┼──────────┼──────────┤")
        for r in results:
            print(f"│ {r['model_name']:<15} │ {r['val_iou']:.4f}   │ {r['val_f1']:.4f}   │")
        print("└─────────────────┴──────────┴──────────┘")

        csv_path = config.RESULTS_DIR / "evaluation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model_name", "val_iou", "val_f1"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    else:
        evaluate_model(args.model, args.checkpoint)


if __name__ == "__main__":
    main()
