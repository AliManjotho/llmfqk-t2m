import torch
import torch.nn.functional as F

class ACAS:
    def compute_a_cas(self, predicted_terms, ground_truth_terms):
        """
        Computes Average Contextual Alignment Score (A-CAS).
        Args:
            predicted_terms (torch.Tensor): Predicted embeddings.
            ground_truth_terms (torch.Tensor): Ground truth embeddings.
        Returns:
            float: A-CAS score.
        """
        similarities = F.cosine_similarity(predicted_terms, ground_truth_terms, dim=1)
        return similarities.mean().item()
