"""
save_val_test_augmented_reader_study_single.py — single-eye counterpart of
``save_val_test_augmented_reader_study_double.py``.

Applies the val/test single-eye augmentation pipeline (inner-boundary-clamped
keypoint-aware center crop + resize) to every eye in a split and saves the
result to ``<paths.result_root_path>/<paths.result_name>/<new_path>``, where
``new_path`` is read from the CSV ``new_path`` column.

Unlike the double-eye script, each row in the single-eye CSV is one eye: the
``relative_path`` carries a ``::R``/``::L`` suffix (stripped to get the file on
disk), and the crop is clamped to the contralateral eye's inner canthus via the
``eye_side`` and ``opposite_eye_inner_x_pct`` columns.

Only the visual steps are applied; Normalize and ToTensorV2 are skipped so the
output is a viewable JPG/PNG.

Usage
-----
    conda run -n dinov3 python data_ted/codes/save_val_test_augmented_reader_study_single.py \\
        -c configs/ted_classification/config_reader_study_single.yaml

    # Override the destination root
    conda run -n dinov3 python data_ted/codes/save_val_test_augmented_reader_study_single.py \\
        -c configs/ted_classification/config_reader_study_single.yaml \\
        paths.result_root_path=/tmp/aug_preview

    # Dump a different split
    conda run -n dinov3 python data_ted/codes/save_val_test_augmented_reader_study_single.py \\
        -c configs/ted_classification/config_reader_study_single.yaml --split val
"""

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import polars as pl
from omegaconf import OmegaConf
from skimage import io
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.augmentation_single_eye import (  # noqa: E402
    single_eye_keypoint_aware_crop,
)

# Match the hardcoded column names used by datasets/classification/dataset_single_eye.py
EYE_SIDE_COL = "eye_side"
INNER_BOUNDARY_COL = "opposite_eye_inner_x_pct"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply single-eye val/test augmentation (inner-boundary-clamped "
            "keypoint-aware center crop + resize) and dump each eye under "
            "<result_root_path>/<result_name>/<new_path>."
        )
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split from cfg.dataset.split_values to dump (default: test).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides (e.g. paths.result_root_path=/tmp/preview).",
    )
    return parser.parse_args()


def load_image(path: Path):
    img = io.imread(str(path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    csv_path = Path(cfg.dataset.csv_path)
    image_root = Path(cfg.dataset.image_path)
    out_root = Path(cfg.paths.result_root_path) / cfg.paths.result_name
    out_root.mkdir(parents=True, exist_ok=True)

    name_col = cfg.dataset.name_col
    split_col = cfg.dataset.split_col
    kx_col = cfg.dataset.keypoint_x_col
    ky_col = cfg.dataset.keypoint_y_col
    split_values = list(cfg.dataset.split_values[args.split])

    height, width = cfg.augmentation.image_size
    val_crop_ratio = float(getattr(cfg.augmentation, "val_crop_ratio", 0.1))
    target_ratio = width / height

    resize = A.Compose(
        [A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)]
    )

    df = pl.read_csv(csv_path).filter(pl.col(split_col).is_in(split_values))
    grouped = df.group_by(name_col).agg(
        pl.col(kx_col).alias("kxs"),
        pl.col(ky_col).alias("kys"),
        pl.col(EYE_SIDE_COL).first().alias(EYE_SIDE_COL),
        pl.col(INNER_BOUNDARY_COL).first().alias(INNER_BOUNDARY_COL),
        pl.col("new_path").first().alias("new_path"),
    )

    print(f"Source CSV : {csv_path}")
    print(f"Image root : {image_root}")
    print(f"Output root: {out_root.resolve()}")
    print(f"Split      : {args.split} -> {split_values}")
    print(f"Crop ratio : {val_crop_ratio} (center mode, kp_w-relative, inner-clamped)")
    print(f"Image size : {height} x {width}")
    print(f"Eyes       : {len(grouped)}")

    n_written = 0
    n_skipped = 0
    for row in tqdm(
        grouped.iter_rows(named=True),
        total=len(grouped),
        desc=f"Saving {args.split}",
    ):
        rel = row[name_col]
        # Strip the ::R/::L suffix to get the actual file path on disk
        src = image_root / rel.split("::", 1)[0]
        dst = out_root / row["new_path"]

        if not src.exists():
            tqdm.write(f"[skip] missing source: {src}")
            n_skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        img = load_image(src)
        keypoints = list(zip(row["kxs"], row["kys"]))
        cropped, _ = single_eye_keypoint_aware_crop(
            img,
            keypoints,
            eye_side=row[EYE_SIDE_COL],
            inner_boundary_x_pct=row[INNER_BOUNDARY_COL],
            padding_ratio=val_crop_ratio,
            mode="center",
            aspect_ratio_range=(target_ratio, target_ratio),
        )
        out_img = resize(image=cropped)["image"]
        # skimage.io returns RGB; cv2.imwrite expects BGR.
        cv2.imwrite(str(dst), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        n_written += 1

    print(f"\nDone. Wrote {n_written} images to {out_root.resolve()}")
    if n_skipped:
        print(f"Skipped {n_skipped} missing source images.")


if __name__ == "__main__":
    main()
