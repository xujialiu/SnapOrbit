# loss/keypoints/mse.py
"""Masked MSE loss module for keypoint regression."""

import torch
import torch.nn as nn


class KeypointMSELoss(nn.Module):
    """Masked Mean Squared Error loss for keypoint regression.

    Ignores NaN values in targets, computing loss only on valid keypoints.
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, **kwargs):
        """
        Compute masked MSE loss.

        Args:
            outputs: Model outputs, shape [batch_size, num_keypoints, 2]
            targets: Ground truth values, shape [batch_size, num_keypoints, 2]
                     NaN values indicate missing/invalid keypoints
            **kwargs: Ignored (for interface compatibility)

        Returns:
            dict with keys:
                - loss: scalar loss value (averaged over valid entries)
                - pred: raw outputs
        """
        # Create mask: True where target is valid (not NaN)
        valid_mask = ~torch.isnan(targets)

        # Replace NaN with 0 to avoid NaN in loss computation
        targets_clean = torch.where(valid_mask, targets, torch.zeros_like(targets))

        # Compute squared error
        squared_error = (outputs - targets_clean) ** 2

        # Apply mask
        masked_error = squared_error * valid_mask.float()

        # Average over valid entries only
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss = masked_error.sum() / num_valid
        else:
            # Maintain gradient connection to outputs
            loss = (outputs * 0.0).sum()

        return {"loss": loss, "pred": outputs}


def create_loss(cfg, **kwargs):
    """
    Factory function to create KeypointMSELoss.

    Args:
        cfg: Loss config (OmegaConf node)
        **kwargs: Ignored

    Returns:
        KeypointMSELoss instance
    """
    return KeypointMSELoss()
