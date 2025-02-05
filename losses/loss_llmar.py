import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMARLoss(nn.Module):
    """
    Loss function for the LLM-Guided Ambiguity Resolver (LLM-AR).
    """

    def __init__(self):
        super(LLMARLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted_context, target_context):
        """
        Computes the LLM-AR loss.
        Args:
            predicted_context (torch.Tensor): Predicted contextual term embeddings (batch_size, embed_dim).
            target_context (torch.Tensor): Ground truth embeddings (batch_size, embed_dim).
        Returns:
            torch.Tensor: Semantic alignment loss.
        """
        similarity_score = self.cosine_similarity(predicted_context, target_context)
        loss = 1 - similarity_score.mean()  # Maximize similarity
        return loss
