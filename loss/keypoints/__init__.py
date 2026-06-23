# loss/keypoints/__init__.py
from .mse import KeypointMSELoss, create_loss

__all__ = ["KeypointMSELoss", "create_loss"]
