import datetime
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import inf


class TeeOutput:
    """Redirect stdout to both console and a log file, with timestamps."""

    def __init__(self, filepath, stream=None):
        self.stream = stream or sys.stdout
        self.file = open(filepath, "a", encoding="utf-8")
        self._at_line_start = True

    def write(self, message):
        if not message:
            return
        lines = message.split("\n")
        parts = []
        for i, line in enumerate(lines):
            if i > 0:
                parts.append("\n")
                self._at_line_start = True
            if line:
                if self._at_line_start:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    parts.append(f"[{ts}] {line}")
                else:
                    parts.append(line)
                self._at_line_start = False
        output = "".join(parts)
        self.stream.write(output)
        self.file.write(output)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def setup_tee_output(log_dir):
    """Redirect stdout to also write to log_dir/log.log."""
    log_path = Path(log_dir) / "log.log"
    sys.stdout = TeeOutput(log_path)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        n = len(iterable)
        per_it = f"{total_time / n:.4f}" if n > 0 else "N/A"
        print(f"{header} Total time: {total_time_str} ({per_it} s / it)")


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def cleanup_old_checkpoints(
    output_dir: Path, pattern: str, keep: int, exclude_pattern: str = None
):
    """Remove old checkpoints, keeping only the N most recent by epoch number."""
    files = list(output_dir.glob(pattern))
    if exclude_pattern:
        exclude = set(output_dir.glob(exclude_pattern))
        files = [f for f in files if f not in exclude]
    if len(files) <= keep:
        return

    def get_epoch(f: Path) -> int:
        # checkpoint_best_5.pth -> 5, checkpoint_10.pth -> 10
        return int(f.stem.split("_")[-1])

    files.sort(key=get_epoch)

    for f in files[:-keep]:
        f.unlink()
        print(f"Removed old checkpoint: {f}")


class BestCheckpointTracker:
    """Track and maintain the top-N best checkpoints by metric value.

    Maintains a sorted queue of (metric_value, epoch) entries. When a new metric
    qualifies for the top-N, the checkpoint is saved and the worst entry is evicted
    (its file deleted from disk).

    On initialization, reconstructs the queue from existing checkpoint files on disk
    and metric values from training_history.csv for resume support.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int,
        mode: str,
        log_dir: str,
        start_epoch: int,
        monitor_metrics_history: list[float],
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.mode = mode
        # entries sorted worst-first: for "max" mode ascending, for "min" mode descending
        self.entries: list[tuple[float, int]] = []

        if start_epoch > 0:
            self._rebuild(monitor_metrics_history)

    def _rebuild(self, monitor_metrics_history: list[float]) -> None:
        """Rebuild queue from checkpoint files on disk + history metrics."""
        existing_files = list(self.checkpoint_dir.glob("checkpoint_best_*.pth"))
        if not existing_files:
            return

        saved_epochs = set()
        for f in existing_files:
            try:
                saved_epochs.add(int(f.stem.split("_")[-1]))
            except ValueError:
                continue

        for epoch in saved_epochs:
            if epoch < len(monitor_metrics_history):
                metric = monitor_metrics_history[epoch]
                self.entries.append((metric, epoch))

        self._sort_entries()
        print(f"BestCheckpointTracker: rebuilt {len(self.entries)} entries from disk")
        for metric, epoch in self.entries:
            print(f"  epoch {epoch}: {metric}")

    def _sort_entries(self) -> None:
        """Sort entries worst-first (so entries[-1] is the best)."""
        if self.mode == "max":
            self.entries.sort(key=lambda x: x[0])  # ascending: worst (lowest) first
        else:
            self.entries.sort(
                key=lambda x: x[0], reverse=True
            )  # descending: worst (highest) first

    def _is_better(self, value: float, reference: float) -> bool:
        """Check if value is better than reference."""
        if self.mode == "max":
            return value > reference
        return value < reference

    def should_save(self, metric_value: float) -> bool:
        """Check if the metric qualifies for the top-N queue."""
        if self.max_keep <= 0:
            return False
        if len(self.entries) < self.max_keep:
            return True
        # Compare against worst entry (first in sorted list)
        worst_value = self.entries[0][0]
        return self._is_better(metric_value, worst_value)

    def update(self, epoch: int, metric_value: float, save_fn) -> bool:
        """Save checkpoint if metric qualifies for top-N, evict worst if needed.

        Args:
            epoch: Current epoch number.
            metric_value: Current metric value.
            save_fn: Callable that takes (epoch,) and saves the checkpoint file.

        Returns:
            True if a checkpoint was saved.
        """
        if not self.should_save(metric_value):
            return False

        # Save the new checkpoint
        save_fn(epoch)
        self.entries.append((metric_value, epoch))
        self._sort_entries()

        # Evict worst if over capacity
        if len(self.entries) > self.max_keep:
            worst_metric, worst_epoch = self.entries.pop(0)
            worst_path = self.checkpoint_dir / f"checkpoint_best_{worst_epoch}.pth"
            if worst_path.exists():
                worst_path.unlink()
                print(
                    f"Evicted checkpoint epoch {worst_epoch} (metric={worst_metric})"
                )

        return True


def save_model_best(cfg, epoch, model, optimizer, loss_scaler, scheduler=None):
    """Save best checkpoint file. Cleanup is handled by BestCheckpointTracker."""
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    epoch_name = str(epoch)

    if loss_scaler is not None:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_best_{epoch_name}.pth"
        )
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": loss_scaler.state_dict(),
            "args": cfg,
        }
        if scheduler is not None:
            to_save["lr_scheduler"] = scheduler.state_dict()
        torch.save(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=checkpoint_dir, tag="checkpoint_best", client_state=client_state
        )


def save_model(cfg, epoch, model, optimizer, loss_scaler, scheduler=None):
    """Save epoch checkpoint and cleanup old ones."""
    max_keep = cfg.checkpoint.save_epoch_ckpt
    if max_keep <= 0:
        return

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    epoch_name = str(epoch)

    if loss_scaler is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch_name}.pth")
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": loss_scaler.state_dict(),
            "args": cfg,
        }
        if scheduler is not None:
            to_save["lr_scheduler"] = scheduler.state_dict()
        torch.save(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=checkpoint_dir, tag="checkpoint", client_state=client_state
        )

    cleanup_old_checkpoints(
        checkpoint_dir,
        "checkpoint_*.pth",
        max_keep,
        exclude_pattern="checkpoint_best_*.pth",
    )


def find_resume_checkpoint(result_path: Path, checkpoint_dir_name: str) -> Path | None:
    """
    Find the latest epoch checkpoint for auto-resume.

    Looks for checkpoint_*.pth (excluding checkpoint_best_*.pth) in the
    checkpoint directory and returns the one with the highest epoch number.

    Args:
        result_path: Root result directory (e.g. ./results/binary/my_experiment)
        checkpoint_dir_name: Relative name of checkpoint subdirectory (e.g. "checkpoints")

    Returns:
        Path to the latest checkpoint, or None if no valid checkpoint found.
    """
    ckpt_dir = result_path / checkpoint_dir_name
    if not result_path.is_dir() or not ckpt_dir.is_dir():
        return None

    files = list(ckpt_dir.glob("checkpoint_*.pth"))
    # Exclude best checkpoints
    exclude = set(ckpt_dir.glob("checkpoint_best_*.pth"))
    files = [f for f in files if f not in exclude]

    if not files:
        return None

    def get_epoch(f: Path) -> int:
        return int(f.stem.split("_")[-1])

    files.sort(key=get_epoch)
    return files[-1]


def _is_eval_mode(cfg) -> bool:
    """Check whether cfg indicates eval mode (handles both config styles).

    Old style: cfg.eval is a plain bool (True/False).
    New style: cfg.eval is a dict with cfg.eval.enable_eval bool.
    """
    eval_cfg = getattr(cfg, "eval", False)
    if isinstance(eval_cfg, bool):
        return eval_cfg
    return getattr(eval_cfg, "enable_eval", False)


def resume_model(cfg, model, optimizer, loss_scaler, scheduler=None) -> int:
    """Load checkpoint state into model/optimizer/scaler in-place.

    Returns the next epoch index to start training from:
      - 0 if no resume happened (no resume_ckpt_path, eval mode, or checkpoint
        lacks optimizer/epoch).
      - checkpoint["epoch"] + 1 if a full resume happened.
    """
    resume_path = getattr(cfg.paths, "resume_ckpt_path", None)
    if not resume_path:
        return 0
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Resume checkpoint {resume_path}")
    if (
        "optimizer" in checkpoint
        and "epoch" in checkpoint
        and not _is_eval_mode(cfg)
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        has_ckpt_state = "lr_scheduler" in checkpoint
        has_scheduler = scheduler is not None
        if has_ckpt_state != has_scheduler:
            raise RuntimeError(
                f"Cross-mode resume not supported: checkpoint has "
                f"lr_scheduler state = {has_ckpt_state}, current run has "
                f"scheduler = {has_scheduler}. Either match the scheduler "
                f"configuration (cosine vs plateau) or start a fresh run."
            )
        if has_scheduler:
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("With optim & sched!")
        return checkpoint["epoch"] + 1
    return 0


def save_training_history(
    log_dir: str,
    train_losses: list[float],
    val_losses: list[float],
    learning_rates: list[float],
    monitor_metrics: list[float] | None = None,
) -> None:
    """Save training history to CSV for resume support.

    Args:
        log_dir: Directory to save the history CSV.
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        learning_rates: Per-epoch learning rate values.
        monitor_metrics: Per-epoch monitored metric values (for best-N checkpoint tracking).
    """
    import csv

    history_path = Path(log_dir) / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "monitor_metric"])
        for i, (tl, vl, lr) in enumerate(zip(train_losses, val_losses, learning_rates)):
            mm = monitor_metrics[i] if monitor_metrics and i < len(monitor_metrics) else ""
            writer.writerow([i, tl, vl, lr, mm])


def load_training_history(
    log_dir: str, start_epoch: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Load training history from CSV up to start_epoch.

    Args:
        log_dir: Directory containing training_history.csv.
        start_epoch: Only load epochs < start_epoch.

    Returns:
        Tuple of (train_losses, val_losses, learning_rates, monitor_metrics) lists.
    """
    import csv

    history_path = Path(log_dir) / "training_history.csv"
    train_losses: list[float] = []
    val_losses: list[float] = []
    learning_rates: list[float] = []
    monitor_metrics: list[float] = []

    if not history_path.exists():
        return train_losses, val_losses, learning_rates, monitor_metrics

    with open(history_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epoch"]) >= start_epoch:
                break
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            learning_rates.append(float(row["lr"]))
            # Backward compatible: old CSVs may not have monitor_metric
            mm = row.get("monitor_metric", "")
            monitor_metrics.append(float(mm) if mm else 0.0)

    return train_losses, val_losses, learning_rates, monitor_metrics


def plot_training_curves(log_dir, train_losses, val_losses, learning_rates):
    """Save train/val loss and learning rate plots to the log directory."""
    epochs = list(range(len(train_losses)))

    # Train & Val Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train & Val Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(Path(log_dir) / "train_val_loss.png", dpi=150)
    plt.close(fig)

    # Learning Rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, learning_rates)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(Path(log_dir) / "lr.png", dpi=150)
    plt.close(fig)
