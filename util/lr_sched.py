import math

import torch.optim.lr_scheduler as torch_sched


def adjust_learning_rate(optimizer, epoch, cfg):
    """Per-iteration LR update.

    - Warmup (epoch < warmup_epochs): linear ramp, always active.
    - Post-warmup: cosine decay if scheduler=='cosine';
      no-op if scheduler=='plateau' (handled per-epoch by ReduceLROnPlateau).
    """
    scheduler = cfg.optimizer.scheduler
    if scheduler not in ("cosine", "plateau"):
        raise ValueError(f"Unknown optimizer.scheduler: {scheduler!r}")

    if epoch < cfg.optimizer.warmup_epochs:
        lr = cfg.optimizer.lr * epoch / cfg.optimizer.warmup_epochs
    elif scheduler == "plateau":
        return None
    else:  # cosine
        lr = cfg.optimizer.min_lr + (cfg.optimizer.lr - cfg.optimizer.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - cfg.optimizer.warmup_epochs)
                / (cfg.training.epochs - cfg.optimizer.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def create_plateau_scheduler(cfg, optimizer):
    """Return a ReduceLROnPlateau if cfg.optimizer.scheduler == 'plateau', else None.

    Reads the following keys under ``cfg.optimizer.plateau`` (defaults match
    ``torch.optim.lr_scheduler.ReduceLROnPlateau`` where applicable):

    - ``monitor_mode``:   "min" | "max"            (required)
    - ``factor``:         float in (0, 1)           (required; e.g. 0.5)
    - ``patience``:       int                       (required; e.g. 3)
    - ``threshold``:      float, default 1e-4
    - ``threshold_mode``: "rel" | "abs", default "rel"
    - ``cooldown``:       int, default 0
    - ``eps``:            float, default 1e-8

    ``min_lr`` is taken from ``cfg.optimizer.min_lr`` (shared with cosine).
    """
    if cfg.optimizer.scheduler != "plateau":
        return None
    p = cfg.optimizer.plateau
    return torch_sched.ReduceLROnPlateau(
        optimizer,
        mode=p.monitor_mode,
        factor=p.factor,
        patience=p.patience,
        threshold=p.get("threshold", 1e-4),
        threshold_mode=p.get("threshold_mode", "rel"),
        cooldown=p.get("cooldown", 0),
        eps=p.get("eps", 1e-8),
        min_lr=cfg.optimizer.min_lr,
    )
