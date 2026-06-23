import random
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def keypoint_aware_crop(
    image,
    keypoints,
    padding_ratio=0.1,
    mode="random",
    aspect_ratio_range=(0.8, 1.2),
    keypoint_jitter_std=0.0,
):
    """
    Crop image around the bounding box of all keypoints.

    v3 vs v2: in val/test mode (`mode="center"`), `padding_ratio` is now a
    fraction of the HORIZONTAL span of all keypoints (kp_w), applied
    isotropically on all four sides. For double-eye images with 4 canthi,
    kp_w is effectively the inter-outer-canthal distance.
    v2 used image width for x-padding and image height for y-padding, which
    was decoupled from face/eye size.

    Args:
        image: HxWxC numpy array
        keypoints: list of (x_pct, y_pct) in 0-100 range
        padding_ratio:
            - train (mode="random"): minimum padding floor as fraction of
              keypoint bbox size (kp_w for x, kp_h for y); actual padding is
              uniformly sampled between this floor and the available room
              to the image edge.
            - val/test (mode="center"): padding on EACH of the four sides as
              fraction of the horizontal span of keypoints (kp_w). Example:
              kp_w = 200 px (inter-outer-canthal distance), padding_ratio =
              0.05 -> 10 px on left, right, top, bottom.
        mode: "random" for train, "center" for val/test
        aspect_ratio_range: (min, max) allowed crop_w / crop_h ratio
        keypoint_jitter_std: stdev of Gaussian noise added to each keypoint
                             (in 0-100 percentage units; only applied in
                             random mode). 0.0 disables jitter.

    Returns:
        (cropped image, (x1, y1, x2, y2)) tuple
    """
    h, w = image.shape[:2]

    if mode == "random" and keypoint_jitter_std > 0:
        keypoints = [
            (
                kp[0] + random.gauss(0, keypoint_jitter_std),
                kp[1] + random.gauss(0, keypoint_jitter_std),
            )
            for kp in keypoints
        ]

    xs = [kp[0] / 100.0 * w for kp in keypoints]
    ys = [kp[1] / 100.0 * h for kp in keypoints]

    kp_x_min, kp_x_max = min(xs), max(xs)
    kp_y_min, kp_y_max = min(ys), max(ys)

    kp_w = kp_x_max - kp_x_min
    kp_h = kp_y_max - kp_y_min

    if mode == "random":
        min_pad_x = padding_ratio * kp_w
        min_pad_y = padding_ratio * kp_h

        pad_left = random.uniform(min_pad_x, max(min_pad_x, kp_x_min))
        pad_right = random.uniform(min_pad_x, max(min_pad_x, w - kp_x_max))
        pad_top = random.uniform(min_pad_y, max(min_pad_y, kp_y_min))
        pad_bottom = random.uniform(min_pad_y, max(min_pad_y, h - kp_y_max))
    else:
        pad = padding_ratio * kp_w
        pad_left = pad
        pad_right = pad
        pad_top = pad
        pad_bottom = pad

    x1 = int(max(0, kp_x_min - pad_left))
    y1 = int(max(0, kp_y_min - pad_top))
    x2 = int(min(w, kp_x_max + pad_right))
    y2 = int(min(h, kp_y_max + pad_bottom))

    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)

    crop_w = x2 - x1
    crop_h = y2 - y1
    ratio_min, ratio_max = aspect_ratio_range

    if crop_h > 0:
        ratio = crop_w / crop_h
        if ratio < ratio_min:
            target_w = int(crop_h * ratio_min)
            expand = target_w - crop_w
            expand_left = expand // 2
            expand_right = expand - expand_left
            x1 = max(0, x1 - expand_left)
            x2 = min(w, x2 + expand_right)
        elif ratio > ratio_max:
            target_h = int(crop_w / ratio_max)
            expand = target_h - crop_h
            expand_top = expand // 2
            expand_bottom = expand - expand_top
            y1 = max(0, y1 - expand_top)
            y2 = min(h, y2 + expand_bottom)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


class KeypointAwareTransform:
    """
    Wrapper that first crops around keypoints, then applies albumentations pipeline.
    Called as: transform(image=img, keypoints=keypoints)
    Returns: dict with "image" key (tensor)
    """

    def __init__(
        self,
        crop_mode,
        crop_padding_ratio,
        aspect_ratio_range,
        album_pipeline,
        pre_crop_pipeline=None,
        keypoint_jitter_std=0.0,
    ):
        self.crop_mode = crop_mode
        self.crop_padding_ratio = crop_padding_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.album_pipeline = album_pipeline
        self.pre_crop_pipeline = pre_crop_pipeline
        self.keypoint_jitter_std = keypoint_jitter_std

    def __call__(self, image, keypoints):
        if self.pre_crop_pipeline is not None:
            image = self.pre_crop_pipeline(image=image)["image"]
        cropped, _ = keypoint_aware_crop(
            image,
            keypoints,
            padding_ratio=self.crop_padding_ratio,
            mode=self.crop_mode,
            aspect_ratio_range=self.aspect_ratio_range,
            keypoint_jitter_std=self.keypoint_jitter_std,
        )
        return self.album_pipeline(image=cropped)


def build_transform(is_train, cfg):
    """
    Build keypoint-aware transform pipeline (v3) for classification.

    v3 vs v2:
      - val/test `val_crop_ratio` is now a fraction of the horizontal span of
        keypoints (kp_w = max_x - min_x), applied equally on all four sides,
        instead of fraction of image width (for x) and image height (for y).
        For double-eye images with 4 canthi, kp_w is the inter-outer-canthal
        distance, so padding becomes anatomy-aware and isotropic.
      - train behavior is unchanged.

    Args:
        is_train: "train", "val", or "test"
        cfg: Full config object

    Returns:
        KeypointAwareTransform callable
    """
    print(f"{is_train} input size: {cfg.augmentation.image_size}")

    mean = cfg.augmentation.mean
    std = cfg.augmentation.std
    height, width = cfg.augmentation.image_size

    target_ratio = width / height
    aspect_tolerance = tuple(getattr(cfg.augmentation, "crop_aspect_ratio", [0.8, 1.2]))
    train_aspect_range = (target_ratio * aspect_tolerance[0], target_ratio * aspect_tolerance[1])

    if is_train == "train":
        crop_padding_ratio = getattr(cfg.augmentation, "crop_padding_ratio", 0.1)
        rotate_limit = getattr(cfg.augmentation, "rotate_limit", 45)
        keypoint_jitter_std = getattr(cfg.augmentation, "keypoint_jitter_std", 0.0)

        pre_crop_pipeline = A.Compose(
            [
                A.Rotate(
                    limit=rotate_limit,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.7,
                ),
            ]
        )

        album_pipeline = A.Compose(
            [
                A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.5,
                ),
                A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ImageCompression(quality_range=(50, 100), p=0.5),
                A.MedianBlur(p=0.2),
                A.GaussNoise(std_range=(0.04, 0.10), p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        return KeypointAwareTransform(
            crop_mode="random",
            crop_padding_ratio=crop_padding_ratio,
            aspect_ratio_range=train_aspect_range,
            album_pipeline=album_pipeline,
            pre_crop_pipeline=pre_crop_pipeline,
            keypoint_jitter_std=keypoint_jitter_std,
        )
    else:
        val_crop_ratio = getattr(cfg.augmentation, "val_crop_ratio", 0.1)

        album_pipeline = A.Compose(
            [
                A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        return KeypointAwareTransform(
            crop_mode="center",
            crop_padding_ratio=val_crop_ratio,
            aspect_ratio_range=(target_ratio, target_ratio),
            album_pipeline=album_pipeline,
        )
