import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GlacialLakeDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform=None, duplicate: bool = False):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform
        self.n_original = len(image_paths)
        # duplicate: adds a deterministic horizontal-flip copy of every sample as an
        # extra training example (custom data duplication operation per Xue et al.)
        self.duplicate = duplicate

    def __len__(self):
        return self.n_original * 2 if self.duplicate else self.n_original

    def __getitem__(self, idx):
        is_dup = self.duplicate and idx >= self.n_original
        real_idx = idx - self.n_original if is_dup else idx

        image = np.array(Image.open(self.image_paths[real_idx]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_paths[real_idx]).convert("L"), dtype=np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if is_dup:
            # deterministic horizontal flip — creates a distinct copy, not a random one
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # (3, H, W) float32
            mask = augmented["mask"]     # (H, W) float32
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)

        # .clone() ensures tensors own their storage — required on Windows
        # when num_workers > 0, because ToTensorV2 produces tensors backed
        # by the numpy array's memory which cannot be resized during collation.
        return image.clone(), mask.clone()


def get_dataloaders(image_dir, mask_dir, batch_size, image_size,
                    train_split, num_workers, pin_memory, seed):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    mask_paths = []
    for img_path in image_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            candidates = list(mask_dir.glob(f"{img_path.stem}*"))
            if not candidates:
                raise FileNotFoundError(
                    f"Mask not found for image '{img_path.name}'. "
                    f"Expected mask at '{mask_path}'"
                )
            mask_path = candidates[0]
        mask_paths.append(mask_path)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        train_size=train_split,
        random_state=seed,
        shuffle=True
    )

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

    # duplicate=True applies the custom data duplication operation on the training set only
    train_dataset = GlacialLakeDataset(train_imgs, train_masks, transform=train_transform, duplicate=True)
    val_dataset = GlacialLakeDataset(val_imgs, val_masks, transform=val_transform, duplicate=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader
