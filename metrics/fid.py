import torch
import torch.nn.functional as F

class FréchetInceptionDistance:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def compute_fid(self, real_features, generated_features):
        """
        Computes Fréchet Inception Distance (FID)
        Args:
            real_features (torch.Tensor): Features of real motion data.
            generated_features (torch.Tensor): Features of generated motion data.
        Returns:
            float: FID score.
        """
        mu_real = torch.mean(real_features, dim=0)
        mu_gen = torch.mean(generated_features, dim=0)
        sigma_real = torch.cov(real_features.T)
        sigma_gen = torch.cov(generated_features.T)

        diff = mu_real - mu_gen
        cov_sqrt, _ = torch.linalg.eigh(sigma_real @ sigma_gen)
        fid = diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * torch.sqrt(cov_sqrt))
        
        return fid.item()
