import torch

class RPrecision:
    def __init__(self, k=1):
        self.k = k

    def compute_r_precision(self, text_features, motion_features):
        """
        Computes R-Precision metric.
        Args:
            text_features (torch.Tensor): Encoded text features.
            motion_features (torch.Tensor): Generated motion features.
        Returns:
            float: R-Precision score.
        """
        distances = torch.cdist(motion_features, text_features, p=2)
        top_k_indices = torch.topk(-distances, self.k, dim=1).indices
        correct = sum([i in top_k_indices[i] for i in range(text_features.size(0))])
        return correct / text_features.size(0)
