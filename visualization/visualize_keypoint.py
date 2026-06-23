"""Visualize keypoint predictions: eval a trained model and draw predicted (+GT)
canthi onto the original images.

Usage:
    python visualization/visualize_keypoint.py \
        -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints/config_vis_kp.yaml

Reuses the SAME model/dataset code paths as train_keypoints.py (create_model,
datasets.keypoints.dataset.build_dataset). The chosen split is run through the model
(eval/Resize-only transform), so the normalized prediction maps to the original image
directly via (x*W, y*H).

Outputs PNG overlays to:  {paths.result_root_path}/{paths.result_name}/{paths.vis_path}/

Config keys read (in addition to the usual train_keypoints keys):
    paths.ckpt_path            checkpoint to load (required)
    paths.vis_path             output sub-dir name (default: "vis")
    visualization.split        which split to draw: train|val|test (default: test)
    visualization.max_images   cap number of images (null = all; default: null)
    visualization.draw_gt      also draw ground-truth keypoints (default: true)
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage import io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from util.config import parse_args_and_config  # noqa: E402

SLOT_COLORS = ["#ff3b30", "#ff9500", "#34c759", "#00c7ff"]
SLOT_LABELS = ["OD outer", "OD inner", "OS inner", "OS outer"]


def load_image(path):
    img = io.imread(str(path))  # raw read, matches datasets/keypoints/dataset.py
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def legend_handles(num_kp, draw_gt):
    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLOT_COLORS[k % 4],
               markeredgecolor="k", markersize=9,
               label=SLOT_LABELS[k] if k < len(SLOT_LABELS) else f"kp{k}")
        for k in range(num_kp)
    ]
    h.append(Line2D([0], [0], marker="X", color="k", linestyle="None", markersize=9, label="pred"))
    if draw_gt:
        h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                        markeredgecolor="k", markersize=9, label="GT"))
    return h


def main():
    cfg = parse_args_and_config()

    # Derive head size (same as train_keypoints.py) and sanity-check sizes
    num_kp = len(cfg.dataset.target_cols)
    cfg.dataset.nb_classes = num_kp * 2
    assert list(cfg.augmentation.image_size) == list(cfg.model.input_size), (
        f"image_size {list(cfg.augmentation.image_size)} != input_size {list(cfg.model.input_size)}"
    )

    device = torch.device(cfg.device)

    # Visualization options
    vis = cfg.get("visualization", {}) or {}
    split = vis.get("split", "test")
    max_images = vis.get("max_images", None)
    draw_gt = vis.get("draw_gt", True)

    out_dir = (
        Path(cfg.paths.result_root_path)
        / cfg.paths.result_name
        / cfg.paths.get("vis_path", "vis")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (eval transform -> Resize+Normalize; normalized coords map to original)
    ds_mod = importlib.import_module(cfg.dataset.module)
    dataset = ds_mod.build_dataset(is_train=split, cfg=cfg)
    if len(dataset) == 0:
        print(f"split '{split}' is empty; falling back to 'val'")
        dataset = ds_mod.build_dataset(is_train="val", cfg=cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
    )

    # Model + checkpoint (same loading as train_keypoints.py eval path)
    model = importlib.import_module(cfg.model.module).create_model(cfg)
    if not cfg.paths.get("ckpt_path", None):
        raise ValueError("paths.ckpt_path must be set in the config for visualization.")
    ckpt = torch.load(cfg.paths.ckpt_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded {cfg.paths.ckpt_path} (epoch {ckpt.get('epoch')}); {msg}")
    model.to(device).eval()

    print(f"split='{split}'  n={len(dataset)}  -> writing overlays to {out_dir}")

    handles = legend_handles(num_kp, draw_gt)
    n_done = 0
    dists = []  # mean per-keypoint L2 distance in normalized space
    with torch.no_grad():
        for imgs, paths, targets in loader:
            out = model(imgs.to(device))
            preds = out.view(out.shape[0], num_kp, 2).float().cpu().numpy()
            tgts = targets.numpy()  # [B, K, 2], normalized 0-1, NaN if missing
            for b in range(len(paths)):
                if max_images is not None and n_done >= max_images:
                    break
                path = paths[b]
                pred = np.clip(preds[b], 0.0, 1.0)
                gt = tgts[b]
                img = load_image(path)
                H, W = img.shape[:2]

                valid = ~np.isnan(gt).any(axis=1)
                if valid.any():
                    d = np.sqrt(((pred[valid] - gt[valid]) ** 2).sum(axis=1))  # normalized
                    dists.append(float(d.mean()))
                    err_str = f"  meanL2(norm)={d.mean():.4f}"
                else:
                    err_str = ""

                fig, ax = plt.subplots(figsize=(9, 9 * H / max(W, 1)))
                ax.imshow(img)
                ax.axis("off")
                for k in range(num_kp):
                    c = SLOT_COLORS[k % len(SLOT_COLORS)]
                    px, py = pred[k, 0] * W, pred[k, 1] * H
                    ax.scatter([px], [py], marker="X", s=140, c=c,
                               edgecolors="k", linewidths=1.0, zorder=4)
                    if draw_gt and valid[k]:
                        gx, gy = gt[k, 0] * W, gt[k, 1] * H
                        ax.scatter([gx], [gy], marker="o", s=110, facecolors="none",
                                   edgecolors=c, linewidths=2.0, zorder=4)
                        ax.plot([gx, px], [gy, py], c=c, lw=1.2, alpha=0.7, zorder=3)
                ax.set_title(f"#{n_done:04d}{err_str}", fontsize=10)
                # anchor the legend's TOP just below the axes so it hangs BELOW the image
                # (loc="upper center" + negative y); bbox_extra_artists keeps it unclipped.
                leg = ax.legend(handles=handles, loc="upper center",
                                bbox_to_anchor=(0.5, -0.01), ncol=min(num_kp + 2, 6),
                                fontsize=8, framealpha=0.9)
                fig.savefig(out_dir / f"{n_done:04d}_{Path(path).stem}.png",
                            dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
                plt.close(fig)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done} done")
            if max_images is not None and n_done >= max_images:
                break

    print(f"Saved {n_done} overlays to {out_dir}")
    if dists:
        print(f"mean per-keypoint L2 (normalized) over {len(dists)} imgs: {np.mean(dists):.4f}")


if __name__ == "__main__":
    main()
