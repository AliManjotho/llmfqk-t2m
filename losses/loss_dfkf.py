import torch
import torch.nn as nn
import torch.nn.functional as F

class DFKFLoss(nn.Module):
    """
    Loss function for the D-FKF module.
    """

    def __init__(self):
        super(DFKFLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, predicted_membership, target_membership, predicted_labels, target_labels):
        """
        Computes the D-FKF loss.
        Args:
            predicted_membership (torch.Tensor): Predicted fuzzy membership values (batch_size, num_terms).
            target_membership (torch.Tensor): Ground truth membership values (batch_size, num_terms).
            predicted_labels (torch.Tensor): Predicted linguistic labels (batch_size, num_classes).
            target_labels (torch.Tensor): Ground truth linguistic labels (batch_size,).
        Returns:
            torch.Tensor: Total loss value.
        """
        membership_loss = self.mse_loss(predicted_membership, target_membership)
        classification_loss = self.cross_entropy(predicted_labels, target_labels)
        total_loss = membership_loss + classification_loss
        return total_loss
