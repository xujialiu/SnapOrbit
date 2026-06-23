# augmentations/keypoints/augmentation_ted.py
#
# Keypoint augmentation for the TED 4-canthus task.
#
# Train pipeline order:
#   0. ASPECT-RATIO aug : random horizontal stretch/compress of the WHOLE image (and
#      keypoints), factor ~ U(aspect_ratio[0], aspect_ratio[1]) on width/height
#      (>1 = wider/flatter face, <1 = narrower/taller). This is the ONLY source of
#      aspect distortion — applied up front, before everything else.
#   0b. HFLIP (p=0.5)   : horizontal flip + REVERSE the keypoint order so slot0 stays the
#      leftmost canthus (see flip note below).
#   1. ROTATE           : random rotation (cv2, manual keypoint transform).
#   2. KEYPOINT-SAFE CROP: crop box DERIVED FROM the keypoints, GUARANTEED to contain all
#      of them (per-side margin kept by a random fraction f ~ U(crop_scale_min,
#      crop_scale_max)); then the box is expanded to a (best-effort) SQUARE so the
#      crop->resize step adds NO further aspect change.
#   3. RESIZE to (h, w).
#   4. PHOTOMETRIC + Normalize + ToTensor (image only).
#
# HORIZONTAL FLIP (p=0.5): the 4 slots are fixed by x-order (slot0=leftmost canthus ...
# slot3=rightmost). A raw flip mirrors x so the x-order reverses; we therefore REVERSE
# the keypoint list, which swaps the symmetric canthus pairs 0<->3 (outer<->outer) and
# 1<->2 (inner<->inner), so slot0 stays the leftmost canthus and targets stay consistent.
# (VerticalFlip is NOT used — anatomically meaningless.)
import random

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(is_train, cfg):
    """Build the transform for the TED keypoint task.

    Returns a callable `transform(image=..., keypoints=...) -> {"image", "keypoints"}`.
    Train uses the aspect-aug + keypoint-safe square crop; val/test is a plain resize.
    """
    print(f"{is_train} input size: {cfg.augmentation.image_size}")

    if is_train == "train":
        return KeypointSafeTrainTransform(cfg)

    height, width = cfg.augmentation.image_size
    return A.Compose(
        [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=cfg.augmentation.mean, std=cfg.augmentation.std),
            ToTensorV2(),
        ],
        # keep every keypoint (resize never moves one out of frame)
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class KeypointSafeTrainTransform:
    """Train pipeline: aspect-stretch -> rotate -> keypoint-safe SQUARE crop -> resize
    -> photometric. The crop is forced square so it adds no aspect distortion; the only
    aspect change is the up-front `aspect_ratio` stretch.
    """

    def __init__(self, cfg):
        self.h, self.w = cfg.augmentation.image_size
        self.rotate_limit = getattr(cfg.augmentation, "rotate_limit", 15)
        self.rotate_p = 0.7
        self.scale_min = getattr(cfg.augmentation, "crop_scale_min", 0.1)
        self.scale_max = getattr(cfg.augmentation, "crop_scale_max", 1.0)

        # aspect_ratio: up-front horizontal stretch/compress factor range (width/height).
        asp = getattr(cfg.augmentation, "aspect_ratio", None)
        if asp is not None:
            self.aspect_min, self.aspect_max = float(asp[0]), float(asp[1])
        else:
            self.aspect_min = self.aspect_max = None

        # the crop is forced to this aspect (= square for a square output) so it does
        # not introduce any stretch of its own.
        self.target_ratio = self.w / self.h

        # Photometric + normalize + to-tensor. Image-only (no geometry), so keypoints
        # are untouched and handled manually.
        self.photometric = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.ImageCompression(quality_range=(50, 100), p=0.2),
                A.Normalize(mean=cfg.augmentation.mean, std=cfg.augmentation.std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, keypoints):
        kps = [(float(x), float(y)) for (x, y) in keypoints]

        # ---- Stage 0: aspect-ratio augmentation (stretch/compress the whole image) ----
        if self.aspect_min is not None and kps:
            ar = random.uniform(self.aspect_min, self.aspect_max)
            H0, W0 = image.shape[:2]
            nW = max(1, int(round(W0 * ar)))
            image = cv2.resize(image, (nW, H0), interpolation=cv2.INTER_CUBIC)
            sx = nW / W0
            kps = [(x * sx, y) for (x, y) in kps]

        # ---- Stage 0b: random horizontal flip (p=0.5) ----
        # Flip the image, mirror x, AND reverse the keypoint order so the fixed slots
        # stay consistent (slot0 = leftmost canthus). Reversing swaps the symmetric
        # canthus pairs 0<->3 (outer<->outer) and 1<->2 (inner<->inner); without it a
        # flip would scramble the slot semantics.
        if kps and random.random() < 0.5:
            Wf = image.shape[1]
            image = cv2.flip(image, 1)
            kps = [(Wf - x, y) for (x, y) in kps][::-1]

        # ---- Stage 1: rotate image + keypoints (MANUAL cv2) ----
        # NOTE: A.Rotate with a reflection border_mode REFLECTS keypoints into a 3x3
        # tiling (4 -> 36 copies), corrupting targets. We rotate with cv2 and apply the
        # same affine matrix to the keypoints -> exactly N points.
        if kps and random.random() < self.rotate_p:
            H0, W0 = image.shape[:2]
            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            M = cv2.getRotationMatrix2D((W0 / 2.0, H0 / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (W0, H0),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
            )
            pts = np.asarray(kps, dtype=np.float64)
            pts = pts @ M[:, :2].T + M[:, 2]      # apply affine to each (x, y)
            kps = [(float(x), float(y)) for x, y in pts]
        H, W = image.shape[:2]

        # ---- Stage 2: keypoint-safe crop box ----
        if len(kps) > 0:
            # clamp into the frame (guards the rare case a point rotates past an edge)
            xs = [min(max(float(k[0]), 0.0), W) for k in kps]
            ys = [min(max(float(k[1]), 0.0), H) for k in kps]
            kps = list(zip(xs, ys))
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # per-side margin between the keypoint bbox and the image edge
            left_m, right_m = x_min, W - x_max
            top_m, bottom_m = y_min, H - y_max

            # keypoint keeps a random fraction f of its margin (f=1 -> no crop)
            new_l = x_min - random.uniform(self.scale_min, self.scale_max) * left_m
            new_r = x_max + random.uniform(self.scale_min, self.scale_max) * right_m
            new_t = y_min - random.uniform(self.scale_min, self.scale_max) * top_m
            new_b = y_max + random.uniform(self.scale_min, self.scale_max) * bottom_m

            # round to ints and guarantee the box still encloses the keypoint bbox
            x0, x1 = int(np.floor(x_min)), int(np.ceil(x_max))
            y0, y1 = int(np.floor(y_min)), int(np.ceil(y_max))
            new_l = max(0, min(int(round(new_l)), x0))
            new_t = max(0, min(int(round(new_t)), y0))
            new_r = min(W, max(int(round(new_r)), x1 + 1))
            new_b = min(H, max(int(round(new_b)), y1 + 1))
        else:
            new_l, new_t, new_r, new_b = 0, 0, W, H

        # ---- Stage 2b: expand the crop to (best-effort) SQUARE = target_ratio, so the
        #      crop->resize adds NO aspect change. Expand outward only -> keypoints stay
        #      inside; clamped to the image bounds. ----
        cw_, ch_ = new_r - new_l, new_b - new_t
        if ch_ > 0:
            ratio = cw_ / ch_
            if ratio < self.target_ratio:  # too narrow -> widen
                expand = int(ch_ * self.target_ratio) - cw_
                add_l = min(expand // 2, new_l)
                add_r = min(expand - add_l, W - new_r)
                add_l = min(add_l + (expand - add_l - add_r), new_l)
                new_l -= add_l
                new_r += add_r
            elif ratio > self.target_ratio:  # too wide -> heighten
                expand = int(cw_ / self.target_ratio) - ch_
                add_t = min(expand // 2, new_t)
                add_b = min(expand - add_t, H - new_b)
                add_t = min(add_t + (expand - add_t - add_b), new_t)
                new_t -= add_t
                new_b += add_b

        crop = image[new_t:new_b, new_l:new_r]
        ch, cw = crop.shape[:2]

        # ---- Stage 3: resize crop to (h, w); scale keypoints into the output frame ----
        crop = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        sx, sy = self.w / cw, self.h / ch
        kps_out = [((x - new_l) * sx, (y - new_t) * sy) for (x, y) in kps]

        # ---- Stage 4: photometric + normalize + to-tensor (image only) ----
        out = self.photometric(image=crop)
        return {"image": out["image"], "keypoints": kps_out}
