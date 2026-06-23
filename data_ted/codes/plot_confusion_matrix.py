import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ["Control", "TED", "Other orbital diseases"]


def plot_confusion_matrix(
    csv_path: Path,
    output_path: Path,
    class_names: list[str] = CLASS_NAMES,
    cmap: str = "Blues",
    normalize: str = "true",
) -> None:
    df = pd.read_csv(csv_path)
    y_true = df["ground_truth"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    labels = list(range(len(class_names)))
    cm_count = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize == "true":
            row_sum = cm_count.sum(axis=1, keepdims=True)
            cm_pct = np.divide(
                cm_count, row_sum, out=np.zeros_like(cm_count, dtype=float), where=row_sum != 0
            )
        elif normalize == "pred":
            col_sum = cm_count.sum(axis=0, keepdims=True)
            cm_pct = np.divide(
                cm_count, col_sum, out=np.zeros_like(cm_count, dtype=float), where=col_sum != 0
            )
        elif normalize == "all":
            total = cm_count.sum()
            cm_pct = cm_count / total if total > 0 else np.zeros_like(cm_count, dtype=float)
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.8 * n + 2, 1.8 * n + 1.5))
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", rotation=270, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    norm_title = {"true": "row-normalized", "pred": "column-normalized", "all": "global"}[normalize]
    ax.set_title(f"Confusion matrix ({norm_title})")

    threshold = cm_pct.max() / 2.0 if cm_pct.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_pct[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm_count[i, j]}\n({cm_pct[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrix from a results.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/data_B/xujialiu/projects/super-resolution-classification/projects/ted_v9/"
            "dinov3_huge_448x896_lora_r32a64_d01_dp01_f1_aug_v3_double_weighted-sampler_eval_prospective/"
            "uncertainty_snap/results.csv"
        ),
        help="Path to results.csv with columns ground_truth, prediction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <csv_dir>/confusion_matrix.png",
    )
    parser.add_argument(
        "--normalize",
        choices=["true", "pred", "all"],
        default="true",
        help="Percentage basis: per true label (row), per predicted label (col), or global.",
    )
    parser.add_argument("--cmap", default="Blues", help="Matplotlib colormap.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=CLASS_NAMES,
        help="Class display names, ordered by class index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.csv.parent / "confusion_matrix.png"
    plot_confusion_matrix(
        csv_path=args.csv,
        output_path=output,
        class_names=args.class_names,
        cmap=args.cmap,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
