import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np


def save_prediction_grid(images, masks, predictions, save_path, n_samples=4):
    """
    Saves a matplotlib figure to save_path.
    Shows n_samples rows, each row has 3 columns:
      [Input Image | Ground Truth Mask | Predicted Mask]
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Input", "Ground Truth", "Prediction"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=12, fontweight="bold")

    for row_idx in range(n):
        img = images[row_idx]
        mask = masks[row_idx]
        pred = predictions[row_idx]

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # (3, H, W) -> (H, W, 3)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)

        mask = mask.squeeze()
        pred = pred.squeeze()

        axes[row_idx][0].imshow(img)
        axes[row_idx][1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx][2].imshow(pred, cmap="gray", vmin=0, vmax=1)

        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, save_path):
    """
    history keys: train_loss, val_loss, train_iou, val_iou, train_f1, val_f1
    Saves a figure with 3 subplots: Loss, IoU, F1
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("Loss", "train_loss", "val_loss"),
        ("IoU",  "train_iou",  "val_iou"),
        ("F1",   "train_f1",   "val_f1"),
    ]

    for ax, (title, train_key, val_key) in zip(axes, metrics):
        ax.plot(epochs, history[train_key], label="Train", marker="o", markersize=3)
        ax.plot(epochs, history[val_key],   label="Val",   marker="s", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
