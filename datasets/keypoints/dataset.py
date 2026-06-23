# datasets/keypoints/dataset.py
import importlib
import math
from torch.utils.data import Dataset
import polars as pl
import cv2
from pathlib import Path
from skimage import io
import torch


def build_dataset(is_train, cfg):
    try:
        module = importlib.import_module(cfg.augmentation.module)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Augmentation module '{cfg.augmentation.module}' not found"
        )

    if not hasattr(module, "build_transform"):
        raise AttributeError(
            f"Module '{cfg.augmentation.module}' must define 'build_transform(is_train, cfg)'"
        )

    transform = module.build_transform(is_train, cfg)

    # Resolve split values: map is_train to configured list of values
    split_values = cfg.dataset.split_values
    if is_train != "all":
        resolved_split = list(split_values[is_train])
    else:
        resolved_split = is_train

    dataset = LoadData(
        csv_path=cfg.dataset.csv_path,
        data_path=cfg.dataset.data_path,
        image_col=cfg.dataset.image_col,
        target_cols=cfg.dataset.target_cols,
        split_col=cfg.dataset.split_col,
        is_train=resolved_split,
        transform=transform,
    )
    return dataset


class LoadData(Dataset):
    def __init__(
        self,
        csv_path,
        data_path,
        image_col,
        target_cols,
        split_col,
        is_train="train",
        transform=None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.data_path = Path(data_path)
        self.image_col = image_col
        self.transform = transform
        self.df = pl.read_csv(csv_path)

        # Parse grouped target_cols: [[x1, y1], [x2, y2], ...]
        self.keypoint_pairs = []
        for pair in target_cols:
            if len(pair) != 2:
                raise ValueError(
                    f"Each keypoint must have exactly 2 columns (x, y), got: {pair}"
                )
            self.keypoint_pairs.append((pair[0], pair[1]))

        self.num_keypoints = len(self.keypoint_pairs)

        # Filter by train/val/test split
        if isinstance(is_train, (list, tuple)):
            self.df = self.df.filter(pl.col(split_col).is_in(is_train))
        elif is_train == "all":
            pass
        else:
            raise ValueError("is_train should be 'all' or a list of split values.")

        # Validate all keypoint columns exist
        all_cols = [col for pair in self.keypoint_pairs for col in pair]
        missing_cols = [col for col in all_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Target columns not found in CSV: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.item(index, self.image_col)
        img_path = self.data_path / img_name

        img = io.imread(str(img_path))

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif (img.shape[-1] != 3) or (img.ndim != 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]

        # Extract keypoints: convert normalized (0-1) to pixel coordinates
        # Use list of tuples for Albumentations, None for missing keypoints
        keypoints = []
        keypoint_indices = []  # Track which keypoints are valid

        for i, (x_col, y_col) in enumerate(self.keypoint_pairs):
            x_norm = self.df.item(index, x_col)
            y_norm = self.df.item(index, y_col)

            # Check for NaN (missing keypoints)
            if (
                x_norm is None
                or y_norm is None
                or math.isnan(x_norm)
                or math.isnan(y_norm)
            ):
                # Skip this keypoint (will be marked as NaN in output)
                continue
            else:
                # Convert normalized to pixel coordinates
                keypoints.append((x_norm * w, y_norm * h))
                keypoint_indices.append(i)

        # Apply augmentation
        transformed = self.transform(image=img, keypoints=keypoints)
        img_tensor = transformed["image"]
        transformed_keypoints = transformed["keypoints"]

        # Get output image dimensions (after resize)
        _, out_h, out_w = img_tensor.shape

        # Initialize targets with NaN
        targets = torch.full((self.num_keypoints, 2), float("nan"), dtype=torch.float32)

        # Fill in valid keypoints (convert back to normalized)
        for idx, kp in zip(keypoint_indices, transformed_keypoints):
            x_px, y_px = kp[0], kp[1]
            # Convert pixel to normalized (0-1)
            x_norm = x_px / out_w
            y_norm = y_px / out_h
            targets[idx] = torch.tensor([x_norm, y_norm], dtype=torch.float32)

        return (img_tensor, str(img_path), targets)
