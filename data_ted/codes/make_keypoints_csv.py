"""Build the wide-format 4-keypoint CSV for ted_keypoints.

Source: projects/ted_classification/data_splited.csv (LONG format)
    - one row per keypoint, 4 keypoints per image (relative_path)
    - keypoint_x / keypoint_y are in PERCENT (0-100) of image width / height
    - 10-fold CV assignment in `fold` (per-image, constant across its 4 rows)

The 4 canthus points are assigned to FIXED slots by sorting on keypoint_x
(left -> right across the face):

    slot -> column   anatomy (image-left == patient's RIGHT eye = OD)
    ------------------------------------------------------------------
    0    -> x1,y1    OD outer canthus
    1    -> x2,y2    OD inner canthus
    2    -> x3,y3    OS inner canthus
    3    -> x4,y4    OS outer canthus

Output: projects/ted_keypoints/data_keypoints.csv (WIDE format)
    columns: relative_path, x1,y1, x2,y2, x3,y3, x4,y4, fold, label, diagnosis, eye, pid
    coordinates NORMALIZED to 0-1 (datasets/keypoints/dataset.py expects 0-1).

Usage:
    conda run -n dinov3 python data_ted/codes/make_keypoints_csv.py
"""

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "projects/ted_classification/data_splited.csv"
DST = REPO / "projects/ted_keypoints/data_keypoints.csv"

N_KP = 4


def main() -> None:
    df = pl.read_csv(SRC)
    print(f"loaded {SRC}  rows={df.height}  images={df['relative_path'].n_unique()}")

    # --- transparency: flag images whose ORIGINAL file order != x-order -----------
    # (sorting by x re-orders them; for 2 known images one point has a near-zero x
    #  that breaks the original left->right order. We keep them and just report.)
    file_order = df.with_columns(
        pl.int_range(pl.len()).over("relative_path").alias("_pos")
    )
    nonmono = (
        file_order.group_by("relative_path", maintain_order=True)
        .agg(
            (pl.col("keypoint_x").sort_by("_pos").diff().drop_nulls() < 0)
            .any()
            .alias("nonmono")
        )
        .filter(pl.col("nonmono"))
        .get_column("relative_path")
        .to_list()
    )
    if nonmono:
        print(
            f"WARNING: {len(nonmono)} image(s) had non-ascending file-order x "
            f"(re-sorted by x for slot assignment):"
        )
        for p in nonmono:
            print(f"    {p}")

    # --- sort by x within each image, then collect into fixed slots ---------------
    g = (
        df.sort(["relative_path", "keypoint_x"])
        .group_by("relative_path", maintain_order=True)
        .agg(
            pl.col("keypoint_x").alias("xs"),
            pl.col("keypoint_y").alias("ys"),
            pl.first("fold").alias("fold"),
            pl.first("label").alias("label"),
            pl.first("diagnosis").alias("diagnosis"),
            pl.first("eye").alias("eye"),
            pl.first("pid").alias("pid"),
            pl.len().alias("n_kp"),
        )
    )

    bad = g.filter(pl.col("n_kp") != N_KP)
    if bad.height:
        print(f"WARNING: dropping {bad.height} image(s) with != {N_KP} keypoints:")
        for p in bad.get_column("relative_path").to_list():
            print(f"    {p}")
        g = g.filter(pl.col("n_kp") == N_KP)

    # percent (0-100) -> normalized (0-1), one column per slot
    coord_cols = []
    for i in range(N_KP):
        coord_cols.append((pl.col("xs").list.get(i) / 100.0).alias(f"x{i + 1}"))
        coord_cols.append((pl.col("ys").list.get(i) / 100.0).alias(f"y{i + 1}"))
    g = g.with_columns(coord_cols)

    out_cols = (
        ["relative_path"]
        + [c for i in range(N_KP) for c in (f"x{i + 1}", f"y{i + 1}")]
        + ["fold", "label", "diagnosis", "eye", "pid"]
    )
    out = g.select(out_cols)

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(DST)
    print(f"wrote {DST}  rows={out.height}")

    # sanity: coordinate range + per-fold counts
    xy = [f"x{i + 1}" for i in range(N_KP)] + [f"y{i + 1}" for i in range(N_KP)]
    mn = min(out.select(pl.col(c).min()).item() for c in xy)
    mx = max(out.select(pl.col(c).max()).item() for c in xy)
    print(f"coord range: [{mn:.4f}, {mx:.4f}]  (expected within ~[0, 1])")
    print("per-fold image counts:")
    print(out.group_by("fold").len().sort("fold"))


if __name__ == "__main__":
    main()
