import torch

class Diversity:
    def compute_diversity(self, generated_motions, num_samples=300):
        """
        Computes Diversity metric.
        Args:
            generated_motions (torch.Tensor): Motion sequences.
            num_samples (int): Number of random samples.
        Returns:
            float: Diversity score.
        """
        indices = torch.randperm(generated_motions.size(0))[:num_samples]
        sampled_motions = generated_motions[indices]
        pairwise_distances = torch.cdist(sampled_motions, sampled_motions, p=2)
        return torch.mean(pairwise_distances).item()
