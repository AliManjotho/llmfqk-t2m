import torch

class MultiModalDistance:
    def compute_mm_dist(self, text_features, motion_features):
        """
        Computes MM-Distance.
        Args:
            text_features (torch.Tensor): Text embeddings.
            motion_features (torch.Tensor): Motion embeddings.
        Returns:
            float: MM-Distance score.
        """
        distances = torch.norm(motion_features - text_features, dim=1)
        return torch.mean(distances).item()
