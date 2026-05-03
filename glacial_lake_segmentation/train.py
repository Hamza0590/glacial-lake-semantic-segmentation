import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders
from models import get_model, count_parameters
from utils.metrics import IoUMetric, F1Metric, compute_batch_metrics
from utils.visualization import plot_training_curves


def train_one_model(model_name: str, epochs: int, lr: float, batch_size: int):
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"Device  : {config.DEVICE}")

    train_loader, val_loader = get_dataloaders(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        train_split=config.TRAIN_SPLIT,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        seed=config.RANDOM_SEED,
    )

    model = get_model(model_name)
    print(f"Parameters: {count_parameters(model):,}")

    # Fix 3: weight_decay — paper specifies Adam but is silent on regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    # BCEWithLogitsLoss = numerically stable sigmoid + BCE via log-sum-exp trick;
    # avoids log(0) explosions that occurred with BCELoss + saturated sigmoid outputs
    criterion = torch.nn.BCEWithLogitsLoss()

    # Fix 2: ReduceLROnPlateau — paper specifies lr=0.001 as start; scheduler only
    # reduces it when val IoU plateaus, so it does not contradict the paper
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    # Sanity check: verify Sigmoid was removed (model must output unbounded logits)
    model.eval()
    with torch.no_grad():
        _imgs, _ = next(iter(train_loader))
        _raw = model(_imgs.to(config.DEVICE))
        _prob = torch.sigmoid(_raw)
        print(f"[Sanity] Logit range : [{_raw.min():.3f}, {_raw.max():.3f}]")
        print(f"[Sanity] Sigmoid range: [{_prob.min():.4f}, {_prob.max():.4f}]")
        # If Sigmoid is still inside the model, all raw outputs will be in [0, 1].
        # Unbounded logits must have at least some values outside [0, 1].
        assert not (_raw.min() >= 0.0 and _raw.max() <= 1.0), \
            "Model outputs look like probabilities — Sigmoid may still be in the head"
        print("[Sanity] ✓ Model outputs raw logits (Sigmoid correctly removed)")
    model.train()

    history = {
        "train_loss": [], "val_loss": [],
        "train_iou":  [], "val_iou":  [],
        "train_f1":   [], "val_f1":   [],
    }
    best_val_iou = 0.0
    best_val_f1  = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        iou_acc = IoUMetric()
        f1_acc  = F1Metric()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]", leave=False)
        for images, masks in pbar:
            images = images.to(config.DEVICE)
            masks  = masks.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)                         # raw logits
            loss    = criterion(outputs, masks)             # BCEWithLogitsLoss: stable
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds   = torch.sigmoid(outputs.detach())       # sigmoid for metrics only
            batch_m = compute_batch_metrics(preds, masks)
            running_loss += loss.item()
            iou_acc.update(preds, masks)
            f1_acc.update(preds, masks)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             iou=f"{batch_m['iou']:.4f}",
                             f1=f"{batch_m['f1']:.4f}")

        train_loss = running_loss / len(train_loader)
        train_iou  = iou_acc.compute()
        train_f1   = f1_acc.compute()

        # ----- Validation -----
        model.eval()
        val_iou_acc  = IoUMetric()
        val_f1_acc   = F1Metric()
        val_running  = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images  = images.to(config.DEVICE)
                masks   = masks.to(config.DEVICE)
                outputs = model(images)                     # raw logits
                loss    = criterion(outputs, masks)         # stable loss
                val_running += loss.item()
                preds   = torch.sigmoid(outputs)            # sigmoid for metrics
                val_iou_acc.update(preds, masks)
                val_f1_acc.update(preds, masks)

        val_loss = val_running / len(val_loader)
        val_iou  = val_iou_acc.compute()
        val_f1   = val_f1_acc.compute()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e}")

        # ----- Checkpoint every epoch -----
        ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_epoch{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, ckpt_path)

        # Fix 2: step scheduler on val IoU after every epoch
        scheduler.step(val_iou)

        # ----- Best model -----
        if val_iou > best_val_iou:
            best_val_iou  = val_iou
            best_val_f1   = val_f1
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, config.CHECKPOINT_DIR / f"{model_name}_best.pt")
            print(f"  ✓ New best model saved (val_iou={best_val_iou:.4f})")

    # ----- Post-training -----
    plot_training_curves(history, config.RESULTS_DIR / f"{model_name}_curves.png")

    print("\n╔" + "═"*38 + "╗")
    print(f"║  Training Complete: {model_name:<18}║")
    print(f"║  Best Val IoU : {best_val_iou:.4f}{'':>18}║")
    print(f"║  Best Val F1  : {best_val_f1:.4f}{'':>18}║")
    print(f"║  Best Val Loss: {best_val_loss:.4f}{'':>18}║")
    print("╚" + "═"*38 + "╝")


def main():
    parser = argparse.ArgumentParser(description="Train glacial lake segmentation models")
    parser.add_argument("--model",  type=str, required=True,
                        help="Model name or 'all'. Choices: unet, simple_cnn, aspp_segnet, all")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr",     type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch",  type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    models_to_train = config.MODELS if args.model == "all" else [args.model]
    for model_name in models_to_train:
        train_one_model(model_name, args.epochs, args.lr, args.batch)


if __name__ == "__main__":
    main()
