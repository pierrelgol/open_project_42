import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    Loss for object detection with proper normalization:
    - MSE for bounding box coordinates (x_rel, y_rel, w, h)
    - BCE for objectness confidence with separate handling for obj/noobj cells
    Returns total loss as a single scalar.
    """

    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 5, S, S) - predicted [x_rel, y_rel, w, h, conf]
            target: (batch, 5, S, S) - target [x_rel, y_rel, w, h, conf]
        Returns:
            total_loss: scalar
        """
        batch_size = pred.shape[0]

        pred_flat = pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 5)
        target_flat = target.permute(0, 2, 3, 1).reshape(batch_size, -1, 5)

        obj_mask = target_flat[:, :, 4] == 1
        noobj_mask = target_flat[:, :, 4] == 0

        coord_loss = torch.sum(
            obj_mask
            * (
                (pred_flat[:, :, 0] - target_flat[:, :, 0]) ** 2
                + (pred_flat[:, :, 1] - target_flat[:, :, 1]) ** 2
                + (pred_flat[:, :, 2] - target_flat[:, :, 2]) ** 2
                + (pred_flat[:, :, 3] - target_flat[:, :, 3]) ** 2
            )
        )

        obj_conf_loss = torch.sum(obj_mask * (pred_flat[:, :, 4] - 1) ** 2)
        noobj_conf_loss = torch.sum(noobj_mask * pred_flat[:, :, 4] ** 2)

        num_obj = obj_mask.sum() + 1e-6

        total_loss = (
            self.lambda_coord * coord_loss / num_obj
            + obj_conf_loss / num_obj
            + self.lambda_noobj * noobj_conf_loss / batch_size
        )

        return total_loss


def get_loss():
    """Factory function for DetectionLoss."""
    return DetectionLoss()
