import torch

class AFOU:
    def compute_a_fou(self, membership_functions):
        """
        Computes Average Footprint of Uncertainty (A-FOU).
        Args:
            membership_functions (list of torch.Tensor): Membership function values.
        Returns:
            float: A-FOU score.
        """
        total_uncertainty = 0.0
        for mf in membership_functions:
            overlap_area = torch.min(mf, dim=0).values.sum()
            total_area = mf.sum()
            total_uncertainty += overlap_area / total_area

        return total_uncertainty / len(membership_functions)
