import torch
import torch.nn as nn
import torch.nn.functional as F

class CMDMLoss(nn.Module):
    """
    Loss function for Contextual Motion Diffusion Model (C-MDM).
    """

    def __init__(self, lambda_diff=0.40, lambda_text=0.45, lambda_graph=0.40):
        """
        Args:
            lambda_diff (float): Weight for diffusion loss.
            lambda_text (float): Weight for text alignment loss.
            lambda_graph (float): Weight for graph consistency loss.
        """
        super(CMDMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.lambda_diff = lambda_diff
        self.lambda_text = lambda_text
        self.lambda_graph = lambda_graph

    def forward(self, predicted_noise, actual_noise, predicted_text_embed, ground_truth_text_embed, 
                predicted_graph, ground_truth_graph):
        """
        Computes the total loss for C-MDM.
        Args:
            predicted_noise (torch.Tensor): Noise predicted by the denoiser.
            actual_noise (torch.Tensor): Ground truth noise.
            predicted_text_embed (torch.Tensor): Predicted motion-text embeddings.
            ground_truth_text_embed (torch.Tensor): Ground truth text embeddings.
            predicted_graph (torch.Tensor): Predicted motion graph features.
            ground_truth_graph (torch.Tensor): Ground truth graph features.
        Returns:
            torch.Tensor: Total loss value.
        """
        # Diffusion loss
        loss_diffusion = self.mse_loss(predicted_noise, actual_noise)

        # Text alignment loss
        similarity_score = self.cosine_similarity(predicted_text_embed, ground_truth_text_embed)
        loss_text = 1 - similarity_score.mean()

        # Graph consistency loss
        loss_graph = self.mse_loss(predicted_graph, ground_truth_graph)

        # Weighted sum of losses
        total_loss = (self.lambda_diff * loss_diffusion) + (self.lambda_text * loss_text) + (self.lambda_graph * loss_graph)
        return total_loss
