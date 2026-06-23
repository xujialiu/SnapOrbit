# augmentations/keypoints/augmentation.py
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(is_train, cfg):
    """
    Build albumentations transform pipeline for keypoint detection.

    Args:
        is_train: "train", "val", or "test"
        cfg: Full config object

    Returns:
        Albumentations Compose object with KeypointParams
    """
    print(f"{is_train} input size: {cfg.augmentation.image_size}")

    mean = cfg.augmentation.mean
    std = cfg.augmentation.std
    height, width = cfg.augmentation.image_size

    # Keypoint params: xy format, remove keypoints that go out of bounds
    keypoint_params = A.KeypointParams(
        format="xy",
        remove_invisible=True,
    )

    if is_train == "train":
        # Get augmentation params with defaults
        rotate_limit = getattr(cfg.augmentation, "rotate_limit", 45)
        crop_scale_min = getattr(cfg.augmentation, "crop_scale_min", 0.8)
        crop_scale_max = getattr(cfg.augmentation, "crop_scale_max", 1.0)

        transform = A.Compose(
            [
                # Geometric transforms (keypoints will be transformed)
                A.Rotate(
                    limit=rotate_limit,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7,
                ),
                A.RandomResizedCrop(
                    size=(height, width),
                    scale=(crop_scale_min, crop_scale_max),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Color/intensity transforms (don't affect keypoints)
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.ImageCompression(quality_range=(50, 100), p=0.2),
                # Normalize and convert to tensor
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    else:
        # Eval transform: resize only, no augmentation
        transform = A.Compose(
            [
                A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )

    return transform
