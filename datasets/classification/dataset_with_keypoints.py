import importlib
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
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

    # Resolve split values
    split_values = cfg.dataset.split_values
    if is_train != "all":
        resolved_split = list(split_values[is_train])
    else:
        resolved_split = is_train

    dataset = LoadData(
        csv_path=cfg.dataset.csv_path,
        data_path=cfg.dataset.image_path,
        name_col=cfg.dataset.name_col,
        target_col=cfg.dataset.target_cols,
        keypoint_x_col=cfg.dataset.keypoint_x_col,
        keypoint_y_col=cfg.dataset.keypoint_y_col,
        split_col=cfg.dataset.split_col,
        is_train=resolved_split,
        nb_classes=cfg.dataset.nb_classes,
        transform=transform,
    )

    return dataset


class LoadData(Dataset):
    def __init__(
        self,
        csv_path,
        data_path,
        name_col,
        target_col,
        keypoint_x_col,
        keypoint_y_col,
        split_col,
        is_train="train",
        nb_classes=None,
        transform=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.name_col = name_col
        self.target_col = target_col
        self.keypoint_x_col = keypoint_x_col
        self.keypoint_y_col = keypoint_y_col
        self.transform = transform

        df = pl.read_csv(csv_path)

        # Filter by split before grouping
        if isinstance(is_train, (list, tuple)):
            df = df.filter(pl.col(split_col).is_in(is_train))
        elif is_train == "all":
            pass
        else:
            raise ValueError("is_train should be 'all' or a list of split values.")

        # Group by image: aggregate 4 keypoint rows into 1 record
        self.df = (
            df.group_by(name_col)
            .agg(
                pl.col(keypoint_x_col).alias("keypoints_x"),
                pl.col(keypoint_y_col).alias("keypoints_y"),
                pl.col(target_col).first().alias(target_col),
            )
        )

        # Build class_counts
        counts_df = self.df.group_by(target_col).len().sort(target_col)
        if nb_classes is not None:
            class_counts = np.zeros(nb_classes, dtype=np.int64)
            for row in counts_df.iter_rows():
                label_idx, count = row
                if label_idx < nb_classes:
                    class_counts[label_idx] = count
            self.class_counts = class_counts
        else:
            self.class_counts = counts_df["len"].to_numpy()

        self.labels = self.df[target_col].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.item(index, self.name_col)
        img_path = self.data_path / img_name

        img = io.imread(str(img_path))

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif (img.shape[-1] != 3) or (img.ndim != 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Extract keypoints as list of (x_pct, y_pct) tuples (0-100 range)
        xs = self.df.item(index, "keypoints_x")
        ys = self.df.item(index, "keypoints_y")
        keypoints = list(zip(xs, ys))

        # Transform expects (image, keypoints) -> returns dict with "image"
        result = self.transform(image=img, keypoints=keypoints)
        img_tensor = result["image"]

        label = self.df.item(index, self.target_col)
        return (img_tensor, str(img_path), label)


def get_weighted_sampler(dataset, num_samples=None):
    class_counts = dataset.class_counts

    print(f"Class counts: {class_counts}")

    counts_tensor = torch.tensor(class_counts, dtype=torch.float)
    class_weights = torch.where(
        counts_tensor > 0, 1.0 / counts_tensor, torch.zeros_like(counts_tensor)
    )
    sample_weights = [class_weights[label] for label in dataset.labels]

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples if num_samples is not None else len(sample_weights),
        replacement=True,
    )
    return weighted_sampler
