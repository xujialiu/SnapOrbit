"""
train_keypoints.py - Keypoint regression training/eval entry point

Loads dataset / model / loss / augmentation dynamically from ``cfg`` and runs
a standard AdamW (+ optional LLRD) finetune loop for keypoint regression.
Each keypoint has an (x, y) pair, so the model outputs ``2 * num_keypoints``
values; this is derived from ``cfg.dataset.target_cols`` at startup and
written back into ``cfg.dataset.nb_classes``. Set ``cfg.eval=true`` to run a
single-pass evaluation on the test split (falling back to val if test is
empty) and then exit.

-------------------------------------------------------------------------------
Full config example (YAML, only fields this script reads directly)
-------------------------------------------------------------------------------
Keys below are the ones ``main()`` itself accesses. Dynamically-loaded
submodules (model / dataset / loss / augmentation) read additional keys
under their own sections — consult their docstrings / configs for those.

    # Runtime
    eval: false                         # true -> run eval on test split and exit
    device: "cuda:0"

    # Training loop
    training:
      seed: 1
      epochs: 20
      batch_size: 8
      accum_iter: 1
      num_iterations: null
      num_workers: 8
      pin_mem: true

    # Loss (no class_weights — regression-style)
    loss:
      module: loss.keypoints.mse
      # ... loss-module-specific keys live under cfg.loss ...

    # Model
    model:
      module: models.model_vit
      input_size: [224, 224]            # asserted == augmentation.image_size
      # ... model-module-specific keys (name / lora / enable_fulltune / ...) live here ...

    # Optimizer
    optimizer:
      lr: null                          # null -> blr * (batch_size * accum_iter) / 256
      blr: 1e-3
      weight_decay: 0.05
      layer_decay: 0.75
      clip_grad: null
      scheduler: cosine               # "cosine" | "plateau"
      use_llrd: true
      warmup_epochs: 5                  # linear lr warmup; plateau.step() is skipped until epoch >= warmup_epochs
      min_lr: 1.0e-6                    # lower bound on lr (cosine floor + ReduceLROnPlateau min_lr)

      # Plateau-scheduler subkeys (only read when scheduler == "plateau")
      plateau:
        monitor_metrics: val_loss       # key in val_metrics passed to scheduler.step()
        monitor_mode: min               # "min" | "max"  (forwarded to ReduceLROnPlateau.mode)
        factor: 0.5                     # lr multiplier on plateau
        patience: 3                     # epochs without improvement before reducing
        threshold: 1.0e-4               # optional; default 1e-4
        threshold_mode: rel             # optional; "rel" | "abs", default "rel"
        cooldown: 0                     # optional; default 0
        eps: 1.0e-8                     # optional; default 1e-8
        stop_lr: null                   # optional; break training once max(param_group.lr) < stop_lr

    # Dataset
    dataset:
      module: datasets.keypoints.dataset
      target_cols:                      # list; nb_classes auto-set to 2 * len(target_cols) (x, y per keypoint)
        - kp1
        - kp2
      train_all: false
      # ... dataset-module-specific keys (csv_path, image_col, splits, ...) live here ...

    # Augmentation
    augmentation:
      image_size: [224, 224]            # asserted == model.input_size
      # ... augmentation-module-specific keys live here ...

    # Paths
    paths:
      result_root_path: ./results
      result_name: keypoints_run
      checkpoint_dir: checkpoints
      log_dir: log
      val_dir: val
      test_dir: test
      ckpt_path: null                   # required only when eval=true

    # Early stopping
    early_stopping:
      patience: 0
      monitor_metrics: val_loss
      monitor_mode: min                 # "min" | "max"

    # Best-checkpoint tracker
    checkpoint:
      save_best_ckpt: 10
      monitor_metrics: val_loss
      monitor_mode: min

-------------------------------------------------------------------------------
Derived behaviours
-------------------------------------------------------------------------------
- ``cfg.dataset.nb_classes`` is overwritten in-place to
  ``2 * len(dataset.target_cols)`` (model outputs x, y pair per keypoint).
- Sampler is ``RandomSampler`` only (no weighted option in this script).
- No class weights are computed; ``create_loss`` is called with ``cfg.loss``
  alone.
- Per-epoch ``log.txt`` entry additionally includes ``val_mse`` pulled from
  ``val_metrics`` (keypoint-specific metric).
- Auto-resume / config snapshot / auto-LR / LLRD: same as
  ``train_classification.py`` (see its docstring).
"""

import numpy as np

import time
from pathlib import Path
from datetime import timedelta
from omegaconf import OmegaConf
from util.config import parse_args_and_config, save_config

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune_keypoints import train, evaluate

import importlib


def create_criterion(cfg, device):
    """Dynamically load and instantiate loss module."""
    module_path = cfg.loss.module
    module = importlib.import_module(module_path)

    criterion = module.create_loss(cfg.loss)
    return criterion.to(device)


def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Validate augmentation image_size matches model input_size
    assert list(cfg.augmentation.image_size) == list(cfg.model.input_size), (
        f"Mismatch: augmentation.image_size={list(cfg.augmentation.image_size)} "
        f"!= model.input_size={list(cfg.model.input_size)}"
    )

    # Derive num_keypoints from target_cols
    num_keypoints = len(cfg.dataset.target_cols)
    cfg.dataset.nb_classes = (
        num_keypoints * 2
    )  # Model outputs N*2 values (x, y per keypoint)
    print(f"Number of keypoints: {num_keypoints}")
    print("Dataset config:")
    for key, value in cfg.dataset.items():
        print(f"  {key}: {value}")

    result_path = Path(cfg.paths.result_root_path) / cfg.paths.result_name

    # Auto-resume: detect existing checkpoint (training only, not eval)
    if not cfg.eval:
        resume_ckpt = misc.find_resume_checkpoint(result_path, cfg.paths.checkpoint_dir)
        if resume_ckpt:
            cfg.paths.resume_ckpt_path = str(resume_ckpt)
            print(f"Auto-resume: found checkpoint {resume_ckpt}")

    # Save config for reproducibility (BEFORE modifying paths)
    if not cfg.eval:
        save_config(cfg, str(result_path))

    # Resolve paths (after saving config to preserve original values)
    cfg.paths.checkpoint_dir = str(result_path / cfg.paths.checkpoint_dir)
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    cfg.paths.log_dir = str(result_path / cfg.paths.log_dir)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    misc.setup_tee_output(cfg.paths.log_dir)

    cfg.paths.save_val_path = str(result_path / cfg.paths.val_dir)
    cfg.paths.save_test_path = str(result_path / cfg.paths.test_dir)

    device = torch.device(cfg.device)

    # fix the seed for reproducibility
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dynamic dataset loading
    dataset_module = importlib.import_module(cfg.dataset.module)

    if cfg.dataset.train_all:
        dataset_train = dataset_module.build_dataset(is_train="all", cfg=cfg)
        dataset_val = dataset_module.build_dataset(is_train="all", cfg=cfg)
    else:
        dataset_train = dataset_module.build_dataset(is_train="train", cfg=cfg)
        dataset_val = dataset_module.build_dataset(is_train="val", cfg=cfg)

    if cfg.eval:
        dataset_test = dataset_module.build_dataset(is_train="test", cfg=cfg)
        if len(dataset_test) == 0:
            print(
                "WARNING: Test split is empty. Falling back to val split for evaluation."
            )
            dataset_test = dataset_module.build_dataset(is_train="val", cfg=cfg)

    # Random sampler for keypoints (no weighted sampling)
    num_samples = (
        cfg.training.num_iterations * cfg.training.batch_size
        if cfg.training.num_iterations
        else len(dataset_train)
    )
    if cfg.training.num_iterations:
        sampler_train = torch.utils.data.RandomSampler(
            dataset_train, replacement=True, num_samples=num_samples
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=False)
    print(f"Sampler_train = {sampler_train} (num_samples={num_samples})")

    # create val sampler
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # create test sampler
    if cfg.eval:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # create train and val data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
        drop_last=False,
    )

    if cfg.eval:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_mem,
            drop_last=False,
        )

    # Dynamic model loading
    model_module = importlib.import_module(cfg.model.module)
    model = model_module.create_model(cfg)

    # Eval: load model checkpoint
    if cfg.eval:
        checkpoint = torch.load(
            cfg.paths.ckpt_path, map_location="cpu", weights_only=False
        )
        print(f"Load checkpoint from: {cfg.paths.ckpt_path}")
        checkpoint_model = checkpoint["model"]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = {model}")
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    eff_batch_size = cfg.training.batch_size * cfg.training.accum_iter

    if cfg.optimizer.lr is None:  # only base_lr is specified
        cfg.optimizer.lr = cfg.optimizer.blr * eff_batch_size / 256

    print(f"base lr: {cfg.optimizer.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {cfg.optimizer.lr:.2e}")

    print(f"accumulate grad iterations: {cfg.training.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if cfg.optimizer.use_llrd:
        # Layer-wise learning rate decay (model-specific implementation)
        param_groups = model.get_param_groups_lrd(
            cfg.optimizer.weight_decay,
            cfg.optimizer.layer_decay,
        )
        for group in param_groups:
            group["lr"] = cfg.optimizer.lr * group.pop("lr_scale")
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.optimizer.lr)

    if cfg.optimizer.use_llrd:
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Parameter group {i}: lr = {param_group['lr']}")

    loss_scaler = NativeScaler()

    plateau_scheduler = lr_sched.create_plateau_scheduler(cfg, optimizer)

    # Dynamic loss loading (no class weights for keypoints)
    criterion = create_criterion(cfg, device)
    print(f"Using loss module: {cfg.loss.module}")

    # Resume model from checkpoint (auto-detected or manually specified)
    start_epoch = misc.resume_model(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        scheduler=plateau_scheduler,
    )

    if cfg.eval:
        criterion.eval()
        evaluate(
            cfg=cfg,
            data_loader=data_loader_test,
            model=model,
            device=device,
            epoch=0,
            mode="test",
            num_keypoints=num_keypoints,
            criterion=criterion,
        )

        exit(0)

    print(f"Start training for {cfg.training.epochs} epochs")
    start_time = time.time()
    best_es_value = (
        float("-inf") if cfg.early_stopping.monitor_mode == "max" else float("inf")
    )
    patience_counter = 0
    history_train_loss, history_val_loss, history_lr, history_monitor_metric = misc.load_training_history(
        cfg.paths.log_dir, start_epoch
    )

    best_ckpt_tracker = misc.BestCheckpointTracker(
        checkpoint_dir=cfg.paths.checkpoint_dir,
        max_keep=cfg.checkpoint.save_best_ckpt,
        mode=cfg.checkpoint.monitor_mode,
        log_dir=cfg.paths.log_dir,
        start_epoch=start_epoch,
        monitor_metrics_history=history_monitor_metric,
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        train_stats = train(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            cfg.optimizer.clip_grad,
            cfg=cfg,
        )

        criterion.eval()
        val_metrics = evaluate(
            cfg=cfg,
            data_loader=data_loader_val,
            model=model,
            device=device,
            epoch=epoch,
            mode="val",
            num_keypoints=num_keypoints,
            criterion=criterion,
        )

        criterion.train()

        # plateau step + lr-based early stop (no-op when plateau_scheduler is None)
        should_stop_lr = False
        if plateau_scheduler is not None and epoch >= cfg.optimizer.warmup_epochs:
            plateau_metric = val_metrics[cfg.optimizer.plateau.monitor_metrics]
            plateau_scheduler.step(plateau_metric)

            stop_lr = cfg.optimizer.plateau.get("stop_lr", None)
            if stop_lr is not None:
                current_max_lr = max(pg["lr"] for pg in optimizer.param_groups)
                if current_max_lr < stop_lr:
                    print(
                        f"Stop: max lr {current_max_lr:.2e} < stop_lr "
                        f"{stop_lr:.2e} at epoch {epoch}"
                    )
                    should_stop_lr = True

        if cfg.paths.checkpoint_dir:
            misc.save_model(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                scheduler=plateau_scheduler,
                epoch=epoch,
            )

        # Save best checkpoint (top-N tracking)
        ckpt_metric = val_metrics[cfg.checkpoint.monitor_metrics]
        if cfg.paths.checkpoint_dir:
            best_ckpt_tracker.update(
                epoch=epoch,
                metric_value=ckpt_metric,
                save_fn=lambda e: misc.save_model_best(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    scheduler=plateau_scheduler,
                    epoch=e,
                ),
            )

        history_train_loss.append(train_stats["loss"])
        history_val_loss.append(val_metrics["val_loss"])
        history_lr.append(train_stats["lr"])
        history_monitor_metric.append(ckpt_metric)
        misc.save_training_history(
            cfg.paths.log_dir, history_train_loss, history_val_loss, history_lr, history_monitor_metric
        )
        misc.plot_training_curves(
            cfg.paths.log_dir, history_train_loss, history_val_loss, history_lr
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
            "val_mse": val_metrics["val_mse"],
        }

        if cfg.paths.checkpoint_dir:
            with open(
                Path(cfg.paths.log_dir) / "log.txt",
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(f"{log_stats}\n")

        # Early stopping
        if cfg.early_stopping.patience > 0:
            es_metric = val_metrics[cfg.early_stopping.monitor_metrics]
            es_improved = (
                es_metric > best_es_value
                if cfg.early_stopping.monitor_mode == "max"
                else es_metric < best_es_value
            )
            if es_improved:
                best_es_value = es_metric
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {cfg.early_stopping.patience}"
                )
                if patience_counter >= cfg.early_stopping.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if should_stop_lr:
            break

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    cfg = parse_args_and_config()
    main(cfg)
