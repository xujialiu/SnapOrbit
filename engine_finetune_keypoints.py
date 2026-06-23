# engine_finetune_keypoints.py
import math
import sys
from pathlib import Path
import torch
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from einops import asnumpy
from sklearn.metrics import mean_squared_error, mean_absolute_error

import polars as pl


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    cfg=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = cfg.training.accum_iter
    num_keypoints = len(cfg.dataset.target_cols)

    optimizer.zero_grad()

    for data_iter_step, (samples, _paths, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per iteration lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, cfg
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()  # [batch, num_kp, 2]

        with torch.amp.autocast("cuda"):
            outputs = model(samples)  # [batch, num_kp * 2]
            # Reshape to [batch, num_keypoints, 2]
            batch_size = outputs.shape[0]
            outputs = outputs.view(batch_size, num_keypoints, 2)

            result = criterion(
                outputs,
                targets,
                epoch=epoch,
                max_epochs=cfg.training.epochs,
            )
            loss = result["loss"]

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_pck(predictions, targets, thresholds=[0.05, 0.10, 0.20]):
    """
    Compute Percentage of Correct Keypoints (PCK) at various thresholds.

    Args:
        predictions: numpy array [N_samples, N_keypoints, 2] - normalized 0-1
        targets: numpy array [N_samples, N_keypoints, 2] - normalized 0-1, NaN for missing

    Returns:
        dict with PCK@threshold for each threshold
    """
    # Euclidean distance between prediction and target
    diff = predictions - targets
    distances = np.sqrt(np.sum(diff**2, axis=-1))  # [N, K]

    # Valid mask (not NaN)
    valid_mask = ~np.isnan(distances)

    results = {}
    num_valid = np.sum(valid_mask)

    if num_valid > 0:
        for thresh in thresholds:
            correct = (distances < thresh) & valid_mask
            pck = np.sum(correct) / num_valid * 100
            results[f"PCK@{thresh}"] = pck
    else:
        for thresh in thresholds:
            results[f"PCK@{thresh}"] = 0.0

    return results


@torch.no_grad()
def evaluate(
    cfg, data_loader, model, device, epoch, mode, num_keypoints, criterion=None
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    all_file_paths = []
    all_predictions = []
    all_ground_truths = []
    all_losses = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image = batch[0]
        paths = batch[1]
        target = batch[-1]  # [batch, num_kp, 2]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        # compute output
        with torch.amp.autocast("cuda"):
            output = model(image)  # [batch, num_kp * 2]
            batch_size = output.shape[0]
            output = output.view(batch_size, num_keypoints, 2)

            result = criterion(
                output,
                target,
                epoch=epoch,
                max_epochs=cfg.training.epochs,
            )
            loss = result["loss"]
            pred = result.get("pred", output)

            # Handle both scalar and per-sample loss
            if loss.dim() == 0:
                loss_value = loss.item()
                loss_array = [loss_value] * batch_size
            else:
                loss_array = asnumpy(loss).tolist()
                loss_value = sum(loss_array) / len(loss_array)

            # Collect predictions and ground truths
            pred_np = asnumpy(pred)  # [batch, num_kp, 2]
            target_np = asnumpy(target)  # [batch, num_kp, 2]

            for idx in range(batch_size):
                all_file_paths.append(paths[idx])
                all_predictions.append(pred_np[idx])
                all_ground_truths.append(target_np[idx])
                all_losses.append(loss_array[idx])

        metric_logger.update(loss=loss_value)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)  # [N, num_kp, 2]
    all_ground_truths = np.array(all_ground_truths)  # [N, num_kp, 2]

    # Create valid mask for metrics
    valid_mask = ~np.isnan(all_ground_truths)

    # Compute MSE/MAE only on valid entries
    valid_preds = all_predictions[valid_mask]
    valid_targets = all_ground_truths[valid_mask]

    if len(valid_preds) > 0:
        mse = mean_squared_error(valid_targets, valid_preds)
        mae = mean_absolute_error(valid_targets, valid_preds)
        rmse = np.sqrt(mse)
    else:
        print(f"WARNING: No valid keypoints found for evaluation at epoch {epoch}")
        mse = mae = rmse = 0.0

    # Compute PCK metrics
    pck_thresholds = getattr(cfg.evaluation, "pck_thresholds", [0.05, 0.10, 0.20])
    pck_metrics = compute_pck(all_predictions, all_ground_truths, pck_thresholds)

    output_loss = np.mean(all_losses)

    # Save results
    metrics_dir = Path(
        cfg.paths.save_test_path if mode == "test" else cfg.paths.save_val_path
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Build results dataframe
    target_cols = list(cfg.dataset.target_cols)
    results_dict = {"file_path": all_file_paths}

    for i, pair in enumerate(target_cols):
        x_col, y_col = pair[0], pair[1]
        results_dict[f"ground_truth_{x_col}"] = all_ground_truths[:, i, 0].tolist()
        results_dict[f"ground_truth_{y_col}"] = all_ground_truths[:, i, 1].tolist()
        results_dict[f"pred_{x_col}"] = all_predictions[:, i, 0].tolist()
        results_dict[f"pred_{y_col}"] = all_predictions[:, i, 1].tolist()

    results_dict["loss"] = all_losses

    pl.DataFrame(results_dict).write_csv(str(metrics_dir / f"results_{epoch}.csv"))

    # Write metrics CSV
    metrics_path = metrics_dir / "metrics.csv"
    row_data = {
        "Mode": [mode],
        "Epoch": [epoch],
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "Loss": [output_loss],
    }
    for t in pck_thresholds:
        row_data[f"PCK@{t}"] = [pck_metrics[f"PCK@{t}"]]
    row = pl.DataFrame(row_data)
    if metrics_path.exists():
        existing = pl.read_csv(metrics_path)
        pl.concat([existing, row]).write_csv(metrics_path)
    else:
        row.write_csv(metrics_path)

    # Print metrics
    pck_str = ", ".join(
        [f"PCK@{t}={pck_metrics[f'PCK@{t}']:.1f}%" for t in pck_thresholds]
    )
    print(
        f"{mode} Epoch {epoch}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, "
        f"Loss={output_loss:.6f}, {pck_str}"
    )

    torch.cuda.empty_cache()
    return {
        "val_mse": mse,
        "val_loss": output_loss,
    }
