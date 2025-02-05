import torch
import torch.nn as nn

class FuzzyLinguisticVocabulary:
    """
    Defines fuzzy linguistic vocabulary for distance, angle, and speed attributes.
    Uses Gaussian Membership Functions to determine membership degrees.
    """

    def __init__(self):
        """
        Initializes linguistic terms and Gaussian parameters.
        """
        self.linguistic_terms = {
            "distance": ["zero", "very-small", "slightly-small", "small", 
                         "slightly-large", "large", "very-large"],
            "angle": ["completely-bent-in", "largely-bent-in", "slightly-bent-in", 
                      "straight", "slightly-bent-out", "largely-bent-out", "completely-bent-out"],
            "speed": ["steady", "slower", "slow", "normal", "fast", "faster"]
        }

        # Define mean and variance for Gaussian Membership Functions
        self.gaussian_params = {
            "distance": torch.tensor([0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),  # Example range [0,1]
            "angle": torch.tensor([-90, -60, -30, 0, 30, 60, 90]),  # Example range [-90,90]
            "speed": torch.tensor([0, 0.3, 0.6, 1.0, 1.5, 2.0]),  # Example range [0,2]
        }
        self.sigma = {key: 0.15 * torch.ones_like(values) for key, values in self.gaussian_params.items()}  # Standard deviation

    def compute_membership(self, x, category):
        """
        Computes Gaussian membership values for a given attribute.
        Args:
            x (torch.Tensor): Input crisp value.
            category (str): One of ['distance', 'angle', 'speed'].
        Returns:
            dict: Membership degrees for each linguistic term.
        """
        mu = self.gaussian_params[category]
        sigma = self.sigma[category]
        membership_values = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        # Map membership values to linguistic terms
        term_values = {self.linguistic_terms[category][i]: membership_values[i].item() for i in range(len(mu))}
        return term_values

    def get_fuzzy_term(self, x, category):
        """
        Returns the linguistic term with the highest membership value.
        Args:
            x (torch.Tensor): Input crisp value.
            category (str): One of ['distance', 'angle', 'speed'].
        Returns:
            str: Linguistic term with highest membership.
        """
        membership = self.compute_membership(x, category)
        return max(membership, key=membership.get), membership

