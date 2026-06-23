"""
get_error_samples.py — read a results CSV from an uncertainty/eval run
(``file_path, ground_truth, prediction[, uncertainty]``), keep the rows
where the prediction is wrong, apply the val/test augmentation pipeline
(keypoint-aware center crop + resize), draw a header strip with
``GT: x    Pred: y`` (and ``Unc`` if present), then dump each image to::

    <paths.result_root_path>/<paths.result_name>/<folder_name>/<relative_path>

``folder_name`` defaults to ``error``.

Usage
-----
    conda run -n dinov3 python data_ted/get_error_samples.py \\
        -c projects/ted_v5/.../config_exp.yaml \\
        --result projects/ted_v5/.../uncertainty_snap/results.csv \\
        --folder_name error
"""

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import polars as pl
from omegaconf import OmegaConf
from skimage import io
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.augmentation_with_keypoints import keypoint_aware_crop  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a results.csv to misclassified samples, apply val/test "
            "augmentation, draw a GT/Pred header, and dump each image to "
            "<result_root_path>/<result_name>/<folder_name>/<relative_path>."
        )
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--result", required=True, help="Path to results.csv")
    parser.add_argument(
        "--folder_name",
        default="error",
        help="Subdir name under <result_root_path>/<result_name>/ (default: error).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides (e.g. paths.result_root_path=/tmp/x).",
    )
    return parser.parse_args()


def load_image(path: Path):
    img = io.imread(str(path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def add_header(img_rgb, text, bar_height=60, bg=(40, 40, 40), fg=(255, 255, 255)):
    """Prepend a solid bar with centered text on top of an RGB image."""
    h, w = img_rgb.shape[:2]
    bar = np.full((bar_height, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = (bar_height + th) // 2
    cv2.putText(bar, text, (x, y), font, scale, fg, thickness, cv2.LINE_AA)
    return np.vstack([bar, img_rgb])


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    image_root = Path(cfg.dataset.image_path)
    data_csv = Path(cfg.dataset.csv_path)
    out_root = (
        Path(cfg.paths.result_root_path) / cfg.paths.result_name / args.folder_name
    )
    out_root.mkdir(parents=True, exist_ok=True)

    name_col = cfg.dataset.name_col
    kx_col = cfg.dataset.keypoint_x_col
    ky_col = cfg.dataset.keypoint_y_col

    height, width = cfg.augmentation.image_size
    val_crop_ratio = float(getattr(cfg.augmentation, "val_crop_ratio", 0.1))
    target_ratio = width / height

    resize = A.Compose(
        [A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)]
    )

    # Build relative_path -> (xs, ys) lookup from the dataset's keypoint CSV.
    data_df = pl.read_csv(data_csv).group_by(name_col).agg(
        pl.col(kx_col).alias("kxs"),
        pl.col(ky_col).alias("kys"),
    )
    kp_lookup = {
        row[name_col]: (row["kxs"], row["kys"])
        for row in data_df.iter_rows(named=True)
    }

    # Filter results to misclassified rows only.
    res = pl.read_csv(args.result).filter(
        pl.col("ground_truth") != pl.col("prediction")
    )
    has_unc = "uncertainty" in res.columns

    print(f"Config     : {args.config}")
    print(f"Results    : {args.result}")
    print(f"Image root : {image_root}")
    print(f"Output root: {out_root.resolve()}")
    print(f"Image size : {height} x {width}, val_crop_ratio={val_crop_ratio}")
    print(f"Errors     : {len(res)}")

    n_written = 0
    n_skipped = 0
    for row in tqdm(
        res.iter_rows(named=True), total=len(res), desc=f"Saving {args.folder_name}"
    ):
        abs_path = Path(row["file_path"])
        try:
            rel = abs_path.relative_to(image_root)
        except ValueError:
            tqdm.write(f"[skip] {abs_path} not under {image_root}")
            n_skipped += 1
            continue
        rel_str = str(rel)

        if rel_str not in kp_lookup:
            tqdm.write(f"[skip] no keypoints in {data_csv.name} for {rel_str}")
            n_skipped += 1
            continue
        if not abs_path.exists():
            tqdm.write(f"[skip] missing source: {abs_path}")
            n_skipped += 1
            continue

        img = load_image(abs_path)
        xs, ys = kp_lookup[rel_str]
        keypoints = list(zip(xs, ys))
        cropped, _ = keypoint_aware_crop(
            img,
            keypoints,
            padding_ratio=val_crop_ratio,
            mode="center",
            aspect_ratio_range=(target_ratio, target_ratio),
        )
        out_img = resize(image=cropped)["image"]

        gt = row["ground_truth"]
        pred = row["prediction"]
        if has_unc and row.get("uncertainty") is not None:
            title = f"GT: {gt}    Pred: {pred}    Unc: {row['uncertainty']:.3f}"
        else:
            title = f"GT: {gt}    Pred: {pred}"
        out_img = add_header(out_img, title)

        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        n_written += 1

    print(f"\nDone. Wrote {n_written} error images to {out_root.resolve()}")
    if n_skipped:
        print(f"Skipped {n_skipped} samples.")


if __name__ == "__main__":
    main()
