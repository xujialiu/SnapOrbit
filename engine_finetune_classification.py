import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from einops import asnumpy
from sklearn import metrics

import polars as pl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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

    optimizer.zero_grad()

    for data_iter_step, (samples, _paths, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, cfg
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(samples)
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


@torch.no_grad()
def evaluate(cfg, data_loader, model, device, epoch, mode, num_class, criterion=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    file_paths = []
    output_probs = []  # for AUC calculation
    predictions = []
    ground_truths = []
    probabilities = []
    losses = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image = batch[0]
        paths = batch[1]
        target = batch[-1]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda"):
            output = model(image)

            result = criterion(
                output,
                target,
                epoch=epoch,
                max_epochs=cfg.training.epochs,
            )
            loss = result["loss"]

            data_bach = image.shape[0]
            num_total += data_bach

            pred_decision = output.argmax(dim=-1)
            pred_softmax = F.softmax(output, dim=-1)

            # Handle both scalar and per-sample loss for CSV output
            if loss.dim() == 0:
                # Scalar loss - create per-sample array with same value
                loss_value = loss.item()
                loss_array = [loss_value] * data_bach
            else:
                # Per-sample loss - convert to list
                loss_array = asnumpy(loss).tolist()
                loss_value = sum(loss_array) / len(loss_array)  # Mean for logging

            for idx in range(data_bach):
                file_paths.append(paths[idx])
                output_probs.append(asnumpy(pred_softmax)[idx])
                predictions.append(asnumpy(pred_decision)[idx])
                ground_truths.append(asnumpy(target)[idx])
                probabilities.append(asnumpy(pred_softmax)[idx])
                losses.append(loss_array[idx])

        acc = metrics.accuracy_score(asnumpy(target), asnumpy(pred_decision))

        batch_size = image.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.meters["acc"].update(acc, n=batch_size)

    metrics_dir = Path(
        cfg.paths.save_test_path if mode == "test" else cfg.paths.save_val_path
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # save results
    pl.DataFrame(
        {
            "file_path": file_paths,
            "ground_truths": ground_truths,
            "predictions": predictions,
            "probabilities": [str(x.tolist()) for x in probabilities],
            "loss": losses,
        }
    ).write_csv(str(metrics_dir / f"results_{epoch}.csv"))

    if cfg.dataset.nb_classes > 2:
        try:
            auc_weighted = metrics.roc_auc_score(
                ground_truths, output_probs, multi_class="ovr", average="weighted"
            )
        except ValueError:
            auc_weighted = 0.0

        try:
            auc_macro = metrics.roc_auc_score(
                ground_truths, output_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_macro = 0.0

        accuracy = metrics.accuracy_score(ground_truths, predictions)
        precision = metrics.precision_score(
            ground_truths, predictions, average="weighted"
        )
        sensitivity = metrics.recall_score(
            ground_truths, predictions, average="weighted"
        )
        specificity = weighted_specificity(ground_truths, predictions)
        f1_weighted = metrics.f1_score(ground_truths, predictions, average="weighted")
        f1_macro = metrics.f1_score(ground_truths, predictions, average="macro")
        kappa = metrics.cohen_kappa_score(
            ground_truths, predictions, weights="quadratic"
        )
    else:
        output_probs_binary = [p[1] for p in output_probs]
        try:
            auc_weighted = metrics.roc_auc_score(
                ground_truths,
                output_probs_binary,
            )
        except ValueError:
            auc_weighted = 0.0

        try:
            auc_macro = metrics.roc_auc_score(
                ground_truths,
                output_probs_binary,
            )
        except ValueError:
            auc_macro = 0.0

        accuracy = metrics.accuracy_score(ground_truths, predictions)
        precision = metrics.precision_score(ground_truths, predictions)
        sensitivity = metrics.recall_score(ground_truths, predictions)
        specificity = weighted_specificity(ground_truths, predictions)
        f1_weighted = metrics.f1_score(ground_truths, predictions, average="weighted")
        f1_macro = metrics.f1_score(ground_truths, predictions, average="macro")
        kappa = metrics.cohen_kappa_score(ground_truths, predictions)

    output_loss = np.mean(losses)

    cm_path = str(metrics_dir / f"cm_{epoch}.png")
    plot_cm(ground_truths, predictions, cm_path)

    metrics_path = metrics_dir / "metrics.csv"
    row = pl.DataFrame(
        {
            "Mode": [mode],
            "Epoch": [epoch],
            "AUC_macro": [auc_macro],
            "AUC_weighted": [auc_weighted],
            "F1_macro": [f1_macro],
            "F1_weighted": [f1_weighted],
            "Kappa": [kappa],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Sensitivity": [sensitivity],
            "Specificity": [specificity],
            "Loss": [output_loss],
        }
    )
    if metrics_path.exists():
        existing = pl.read_csv(metrics_path)
        pl.concat([existing, row]).write_csv(metrics_path)
    else:
        row.write_csv(metrics_path)

    print(
        f"{mode} Epoch {epoch}: AUC macro: {auc_macro}, AUC weighted: {auc_weighted}, F1 macro: {f1_macro}, F1 weighted: {f1_weighted}, Kappa: {kappa}, Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Loss: {output_loss}\n"
    )
    torch.cuda.empty_cache()
    return {
        "val_auc_weighted": auc_weighted,
        "val_auc_macro": auc_macro,
        "val_f1_weighted": f1_weighted,
        "val_f1_macro": f1_macro,
        "val_acc": accuracy,
        "val_loss": output_loss,
    }


def plot_cm(ground_truths, predictions, save_path):
    if len(ground_truths) == 0 or len(predictions) == 0:
        print("WARNING: No predictions to plot confusion matrix. Skipping.")
        return

    cm = confusion_matrix(ground_truths, predictions)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Percentage)")

    fig.savefig(save_path)


def weighted_specificity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]

    specificities = []
    class_weights = []

    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))

        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(specificity)

        class_weights.append(np.sum(y_true == i))

    class_weights = np.array(class_weights) / len(y_true)
    weighted_avg = np.sum(np.array(specificities) * class_weights)

    return weighted_avg
