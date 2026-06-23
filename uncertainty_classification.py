"""
uncertainty_classification.py - Per-sample uncertainty for classification

Runs a trained classification model on the test split (falling back to val if
empty) and writes a per-sample uncertainty value in [0, 1] (0 = fully certain,
1 = fully uncertain) into ``<result_root>/<result_name>/<paths.uncertainty>/results.csv``.

Methods (selected via ``cfg.uncertainty.method``):
    mc_dropout                   - T stochastic forward passes through one
                                   checkpoint using the mc_dropout.MCDropout
                                   wrapper.
    snapshot_ensemble            - One forward pass per user-supplied
                                   checkpoint.
    mc_dropout_snapshot_ensemble - T_mc MC Dropout passes per checkpoint;
                                   total estimators = T_mc * N_snapshots.

All methods produce a (T, B, K) probability tensor that feeds a unified
metric block selected by ``cfg.uncertainty.metric``:
    predictive_entropy   H(mean_p) / log(K)
    mutual_information   (H(mean_p) - mean_t H(p_t)) / log(K)
    variance             sum_k var_t(p_t[k]) / ((K-1)/K), clamped

Predictions / saved probabilities use the mean across T (probability-space
averaging then argmax), matching standard MC Dropout / ensemble convention.

-------------------------------------------------------------------------------
Config schema (only fields read by THIS script; model/dataset/aug come from the
shared training config and are loaded the same way as train_classification.py)
-------------------------------------------------------------------------------

    paths:
      uncertainty: uncertainty           # subdir under <result_root>/<result_name>; default \"uncertainty\"
      ckpt_path: /abs/path/to/single.pth # used by mc_dropout only

    uncertainty:
      method: mc_dropout                 # mc_dropout | snapshot_ensemble | mc_dropout_snapshot_ensemble
      metric: predictive_entropy         # predictive_entropy | mutual_information | variance
      other_params:
        # MC Dropout params (mc_dropout, mc_dropout_snapshot_ensemble)
        num_estimators: 25               # T forward passes per checkpoint
        last_layer: false                # only enable dropout on last module
        on_batch: true                   # repeat batch vs. for-loop
        # Snapshot params (snapshot_ensemble, mc_dropout_snapshot_ensemble) — EXACTLY ONE:
        checkpoint_folder_path: null     # dir to glob '*.pth' from (sorted)
        checkpoint_paths:                # OR explicit list
          - /abs/path/to/checkpoint_best_57.pth

CLI:
    conda run -n dinov3 python uncertainty_classification.py \\
        -c configs/<group>/<run>/config_uncertainty.yaml
"""

import importlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from util.config import parse_args_and_config


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

from mc_dropout import mc_dropout as _mc_dropout
from torch.nn.modules.dropout import _DropoutNd


class MCDropoutRunner:
    """Single checkpoint, T stochastic forward passes via MCDropout wrapper."""

    def __init__(self, model, ckpt_path, num_estimators, last_layer, on_batch, device):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Load message: {msg}")

        # Preflight: list captured nn.Dropout modules and their p values.
        captured = [m for m in model.modules() if isinstance(m, _DropoutNd)]
        nonzero = [m for m in captured if m.p > 0.0]
        print(
            f"MCDropout: {len(captured)} dropout modules captured "
            f"({len(nonzero)} with p>0). p values: "
            f"{sorted({round(float(m.p), 4) for m in captured})}"
        )
        if not nonzero:
            print(
                "WARNING: no nn.Dropout module has p>0; MC Dropout will be "
                "deterministic. Check your model's lora.dropout / drop_rate."
            )

        self.num_estimators = int(num_estimators)
        self.on_batch = bool(on_batch)
        self.device = device
        self.model = _mc_dropout(
            model=model,
            num_estimators=self.num_estimators,
            last_layer=bool(last_layer),
            on_batch=self.on_batch,
        ).to(device)
        self.model.eval()  # MCDropout overrides train() so dropout stays active

    @torch.no_grad()
    def run(self, data_loader):
        """Yield per-batch (paths, ground_truths, probs_TBK) tuples."""
        T = self.num_estimators
        for batch in tqdm(data_loader, desc="MC Dropout"):
            images = batch[0].to(self.device, non_blocking=True)
            paths = batch[1]
            targets = batch[-1]
            B = images.shape[0]

            logits = self.model(images)  # (T*B, K) regardless of on_batch
            K = logits.shape[-1]
            logits = logits.view(T, B, K)
            probs = F.softmax(logits, dim=-1)
            yield paths, targets.cpu().tolist(), probs.cpu().float()


class SnapshotEnsembleRunner:
    """N user-supplied checkpoints; one forward pass per checkpoint."""

    def __init__(self, model, checkpoint_paths, device):
        if not checkpoint_paths:
            raise ValueError("snapshot_ensemble requires non-empty checkpoint_paths")
        for pth in checkpoint_paths:
            if not Path(pth).exists():
                raise FileNotFoundError(f"checkpoint not found: {pth}")
        self.model = model
        self.checkpoint_paths = list(checkpoint_paths)
        self.device = device
        print(f"Snapshot ensemble: {len(self.checkpoint_paths)} checkpoints")

    @torch.no_grad()
    def run(self, data_loader):
        """Yield per-batch (paths, ground_truths, probs_TBK) tuples.

        Strategy: for each checkpoint, run the full dataloader and store
        per-batch probability tensors on CPU. After all snapshots, walk
        through batches and stack across the snapshot dimension.
        """
        T = len(self.checkpoint_paths)
        per_snap_probs = []        # [T] of list-of-batch-Tensors (B_i, K)
        cached_paths = None        # set on first snapshot
        cached_targets = None      # set on first snapshot

        for t, ckpt_path in enumerate(self.checkpoint_paths):
            print(f"[{t+1}/{T}] Loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            msg = self.model.load_state_dict(ckpt["model"], strict=False)
            print(f"  Load message: {msg}")
            self.model.eval()

            batch_probs = []
            batch_paths = [] if cached_paths is None else None
            batch_targets = [] if cached_targets is None else None

            for batch in tqdm(data_loader, desc=f"Snapshot {t+1}/{T}"):
                images = batch[0].to(self.device, non_blocking=True)
                paths = batch[1]
                targets = batch[-1]
                logits = self.model(images)
                probs = F.softmax(logits, dim=-1).cpu().float()
                batch_probs.append(probs)
                if batch_paths is not None:
                    batch_paths.append(list(paths))
                if batch_targets is not None:
                    batch_targets.append(targets.cpu().tolist())

            per_snap_probs.append(batch_probs)
            if cached_paths is None:
                cached_paths = batch_paths
                cached_targets = batch_targets

        # Re-yield per batch with (T, B, K) stack
        num_batches = len(cached_paths)
        for b in range(num_batches):
            stacked = torch.stack(
                [per_snap_probs[t][b] for t in range(T)], dim=0
            )  # (T, B, K)
            yield cached_paths[b], cached_targets[b], stacked


class MCDropoutSnapshotEnsembleRunner:
    """T_mc MC Dropout forward passes per snapshot checkpoint.

    Total estimators = num_estimators * len(checkpoint_paths). The MCDropout
    wrapper is built once over the raw model; subsequent checkpoints reload
    state dict into ``self.model.core_model`` in place — the captured dropout
    module references stay valid because load_state_dict does not replace
    modules, only their parameters.
    """

    def __init__(self, model, checkpoint_paths, num_estimators, last_layer, on_batch, device):
        if not checkpoint_paths:
            raise ValueError(
                "mc_dropout_snapshot_ensemble requires non-empty checkpoint_paths"
            )
        for pth in checkpoint_paths:
            if not Path(pth).exists():
                raise FileNotFoundError(f"checkpoint not found: {pth}")

        captured = [m for m in model.modules() if isinstance(m, _DropoutNd)]
        nonzero = [m for m in captured if m.p > 0.0]
        print(
            f"MCDropout: {len(captured)} dropout modules captured "
            f"({len(nonzero)} with p>0). p values: "
            f"{sorted({round(float(m.p), 4) for m in captured})}"
        )
        if not nonzero:
            print(
                "WARNING: no nn.Dropout module has p>0; MC Dropout will be "
                "deterministic. Check your model's lora.dropout / drop_rate."
            )

        self.num_estimators = int(num_estimators)
        self.checkpoint_paths = list(checkpoint_paths)
        self.device = device
        self.model = _mc_dropout(
            model=model,
            num_estimators=self.num_estimators,
            last_layer=bool(last_layer),
            on_batch=bool(on_batch),
        ).to(device)
        self.model.eval()
        total = self.num_estimators * len(self.checkpoint_paths)
        print(
            f"MC Dropout × Snapshot Ensemble: "
            f"{len(self.checkpoint_paths)} checkpoints × "
            f"{self.num_estimators} MC passes = {total} total estimators"
        )

    @torch.no_grad()
    def run(self, data_loader):
        """Yield per-batch (paths, ground_truths, probs_TBK) with T = T_mc * N."""
        T_mc = self.num_estimators
        N = len(self.checkpoint_paths)
        per_snap_probs = []
        cached_paths = None
        cached_targets = None

        for n, ckpt_path in enumerate(self.checkpoint_paths):
            print(f"[{n+1}/{N}] Loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            msg = self.model.core_model.load_state_dict(ckpt["model"], strict=False)
            print(f"  Load message: {msg}")
            self.model.eval()

            batch_probs = []
            batch_paths = [] if cached_paths is None else None
            batch_targets = [] if cached_targets is None else None

            for batch in tqdm(data_loader, desc=f"Snap {n+1}/{N} (T_mc={T_mc})"):
                images = batch[0].to(self.device, non_blocking=True)
                paths = batch[1]
                targets = batch[-1]
                B = images.shape[0]

                logits = self.model(images)  # (T_mc * B, K)
                K = logits.shape[-1]
                logits = logits.view(T_mc, B, K)
                probs = F.softmax(logits, dim=-1).cpu().float()  # (T_mc, B, K)
                batch_probs.append(probs)
                if batch_paths is not None:
                    batch_paths.append(list(paths))
                if batch_targets is not None:
                    batch_targets.append(targets.cpu().tolist())

            per_snap_probs.append(batch_probs)
            if cached_paths is None:
                cached_paths = batch_paths
                cached_targets = batch_targets

        # Concat snapshots along leading dim -> (T_mc * N, B, K)
        num_batches = len(cached_paths)
        for b in range(num_batches):
            stacked = torch.cat(
                [per_snap_probs[n][b] for n in range(N)], dim=0
            )  # (T_mc * N, B, K)
            yield cached_paths[b], cached_targets[b], stacked


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------

def compute_uncertainty(probs, metric, K, eps=1e-12):
    """Collapse a (T, B, K) probability tensor into a per-sample uncertainty in [0, 1].

    Args:
        probs (Tensor): float tensor of shape (T, B, K), softmax probabilities.
        metric (str): "predictive_entropy" | "mutual_information" | "variance".
        K (int): number of classes (>= 2).
        eps (float): numerical floor for log.

    Returns:
        Tensor of shape (B,), values in [0, 1].
    """
    if probs.dim() != 3:
        raise ValueError(f"expected probs of shape (T, B, K); got {tuple(probs.shape)}")
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    mean_p = probs.mean(dim=0)  # (B, K)

    if metric == "predictive_entropy":
        H = -(mean_p * (mean_p + eps).log()).sum(dim=-1)        # (B,)
        return (H / math.log(K)).clamp(0.0, 1.0)

    if metric == "mutual_information":
        H_mean = -(mean_p * (mean_p + eps).log()).sum(dim=-1)   # (B,)
        H_each = -(probs * (probs + eps).log()).sum(dim=-1).mean(dim=0)  # (B,)
        return ((H_mean - H_each) / math.log(K)).clamp(0.0, 1.0)

    if metric == "variance":
        # sum_k var_t(p_t[k]) / ((K-1)/K), clamped
        # Denom is the Gini-impurity upper bound: 1 - sum_k p̄_k² max'd at p̄=1/K.
        var_per_class_sum = probs.var(dim=0, unbiased=False).sum(dim=-1)  # (B,)
        denom = (K - 1) / K
        return (var_per_class_sum / denom).clamp(0.0, 1.0)

    raise ValueError(
        f"unknown uncertainty metric: {metric!r}. "
        "Choose predictive_entropy | mutual_information | variance."
    )


def _resolve_checkpoint_paths(p, method):
    """Resolve checkpoint paths from other_params for snapshot-based methods.

    Accepts either ``checkpoint_folder_path`` (glob ``*.pth`` sorted) OR
    ``checkpoint_paths`` (explicit list). Raises if both are set, neither is
    set, or the folder contains no .pth files.
    """
    folder = p.get("checkpoint_folder_path", None)
    explicit = p.get("checkpoint_paths", None)
    if folder and explicit:
        raise ValueError(
            f"{method}: specify either checkpoint_folder_path or "
            "checkpoint_paths, not both."
        )
    if folder:
        paths = sorted(str(q) for q in Path(folder).glob("*.pth"))
        if not paths:
            raise FileNotFoundError(
                f"No .pth files found under checkpoint_folder_path: {folder}"
            )
        print(f"Discovered {len(paths)} .pth files under {folder}")
        return paths
    if explicit:
        return list(explicit)
    raise ValueError(
        f"{method}: must set either "
        "uncertainty.other_params.checkpoint_folder_path or checkpoint_paths."
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    class_names,
    cmap="Blues",
    normalize="true",
):
    """Save a counts + percentage confusion matrix PNG next to results.csv.

    Mirrors data_ted/codes/plot_confusion_matrix.py but consumes in-memory
    label lists instead of re-reading the CSV. ``normalize`` controls the
    percentage basis: "true" (row), "pred" (column), or "all" (global).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("WARNING: no predictions to plot confusion matrix; skipping.")
        return

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {output_path}")


def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    assert list(cfg.augmentation.image_size) == list(cfg.model.input_size), (
        f"Mismatch: augmentation.image_size={list(cfg.augmentation.image_size)} "
        f"!= model.input_size={list(cfg.model.input_size)}"
    )

    result_path = Path(cfg.paths.result_root_path) / cfg.paths.result_name
    subdir = cfg.paths.get("uncertainty", "uncertainty")
    output_dir = result_path / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Dataset (test split, fall back to val if test is empty)
    dataset_module = importlib.import_module(cfg.dataset.module)
    dataset_test = dataset_module.build_dataset(is_train="test", cfg=cfg)
    if len(dataset_test) == 0:
        print("WARNING: test split empty - falling back to val split.")
        dataset_test = dataset_module.build_dataset(is_train="val", cfg=cfg)
    print(f"Test dataset size: {len(dataset_test)}")

    # Model (no checkpoint loaded yet - runners do that)
    model_module = importlib.import_module(cfg.model.module)
    model = model_module.create_model(cfg)
    model.to(device)

    sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
        drop_last=False,
    )

    method = cfg.uncertainty.method
    metric = cfg.uncertainty.metric
    K = int(cfg.dataset.nb_classes)
    print(f"method={method}  metric={metric}  K={K}")

    p = OmegaConf.to_container(
        cfg.uncertainty.get("other_params", {}) or {}, resolve=True
    )

    if method == "mc_dropout":
        runner = MCDropoutRunner(
            model=model,
            ckpt_path=cfg.paths.ckpt_path,
            num_estimators=p.get("num_estimators", 25),
            last_layer=p.get("last_layer", False),
            on_batch=p.get("on_batch", True),
            device=device,
        )
    elif method == "snapshot_ensemble":
        runner = SnapshotEnsembleRunner(
            model=model,
            checkpoint_paths=_resolve_checkpoint_paths(p, method),
            device=device,
        )
    elif method == "mc_dropout_snapshot_ensemble":
        runner = MCDropoutSnapshotEnsembleRunner(
            model=model,
            checkpoint_paths=_resolve_checkpoint_paths(p, method),
            num_estimators=p.get("num_estimators", 25),
            last_layer=p.get("last_layer", False),
            on_batch=p.get("on_batch", True),
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown uncertainty.method: {method!r}. "
            "Choose mc_dropout | snapshot_ensemble | mc_dropout_snapshot_ensemble."
        )

    file_paths, ground_truths, predictions, probabilities, uncertainties = (
        [], [], [], [], []
    )
    for paths, gts, probs_TBK in runner.run(data_loader):
        mean_p = probs_TBK.mean(dim=0)                         # (B, K)
        preds = mean_p.argmax(dim=-1).tolist()                 # (B,)
        unc = compute_uncertainty(probs_TBK, metric, K).tolist()  # (B,)

        for i in range(mean_p.shape[0]):
            file_paths.append(paths[i])
            ground_truths.append(int(gts[i]))
            predictions.append(int(preds[i]))
            probabilities.append(str(mean_p[i].tolist()))
            uncertainties.append(float(unc[i]))

    out_csv = output_dir / "results.csv"
    pl.DataFrame(
        {
            "file_path": file_paths,
            "ground_truth": ground_truths,
            "prediction": predictions,
            "probabilities": probabilities,
            "uncertainty": uncertainties,
        }
    ).write_csv(str(out_csv))
    print(f"Wrote {len(file_paths)} rows to {out_csv}")

    class_names = cfg.dataset.get("class_names", None)
    if not class_names or len(class_names) != K:
        class_names = [str(i) for i in range(K)]
    plot_confusion_matrix(
        ground_truths,
        predictions,
        output_dir / "confusion_matrix.png",
        list(class_names),
    )


if __name__ == "__main__":
    cfg = parse_args_and_config()
    main(cfg)
