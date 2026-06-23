# loss/cross_entropy.py
"""Cross-entropy loss module."""

import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with optional class weighting and label smoothing."""

    def __init__(self, weight=None, label_smoothing=0.0):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, outputs, targets, **kwargs):
        """
        Compute cross-entropy loss.

        Args:
            outputs: Model outputs (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            **kwargs: Ignored (for interface compatibility)

        Returns:
            dict with keys:
                - loss: scalar loss value
                - pred: softmax probabilities (eval mode only)
        """
        loss = F.cross_entropy(
            outputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        result = {"loss": loss}

        if not self.training:
            result["pred"] = F.softmax(outputs, dim=-1)

        return result


def create_loss(cfg, class_weights=None, **kwargs):
    """
    Factory function to create CrossEntropyLoss.

    Args:
        cfg: Loss config (OmegaConf node with 'weight' and optional 'label_smoothing')
        class_weights: Precomputed class weights tensor
        **kwargs: Ignored

    Returns:
        CrossEntropyLoss instance
    """
    weight = class_weights if cfg.get("weight", False) else None
    label_smoothing = cfg.get("label_smoothing", 0.0)
    return CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
