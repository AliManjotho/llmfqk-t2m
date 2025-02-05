import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianMembershipFunction(nn.Module):
    """
    Implements a Gaussian Membership Function for fuzzy logic.
    """
    def __init__(self, num_features, num_terms):
        """
        Args:
            num_features (int): Number of input features.
            num_terms (int): Number of fuzzy terms per feature.
        """
        super(GaussianMembershipFunction, self).__init__()
        self.mu = nn.Parameter(torch.randn(num_features, num_terms))  # Mean
        self.sigma = nn.Parameter(torch.abs(torch.randn(num_features, num_terms)))  # Standard deviation

    def forward(self, x):
        """
        Computes the membership degree for each input value.
        Args:
            x (torch.Tensor): Input feature tensor (batch_size, num_features).
        Returns:
            torch.Tensor: Membership values (batch_size, num_features, num_terms).
        """
        x = x.unsqueeze(-1)  # Expand for broadcasting
        membership = torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        return membership


class FuzzyInferenceModule(nn.Module):
    """
    Implements fuzzy rule inference using Gaussian Membership Functions.
    """
    def __init__(self, num_features, num_terms, num_rules):
        """
        Args:
            num_features (int): Number of input features.
            num_terms (int): Number of fuzzy terms per feature.
            num_rules (int): Number of fuzzy rules.
        """
        super(FuzzyInferenceModule, self).__init__()
        self.membership_function = GaussianMembershipFunction(num_features, num_terms)
        self.rule_weights = nn.Parameter(torch.randn(num_terms, num_rules))  # Rule weight matrix

    def forward(self, x):
        """
        Applies fuzzy rules to input values.
        Args:
            x (torch.Tensor): Input feature tensor (batch_size, num_features).
        Returns:
            torch.Tensor: Fuzzy output (batch_size, num_rules).
        """
        membership_values = self.membership_function(x)  # (batch_size, num_features, num_terms)
        rule_outputs = torch.matmul(membership_values, self.rule_weights)  # Apply fuzzy rules
        return torch.sum(rule_outputs, dim=1)  # Sum across features


class DeepFeatureExtractor(nn.Module):
    """
    Implements a neural network to learn deep motion representations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
        """
        super(DeepFeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Extracts deep features from input data.
        Args:
            x (torch.Tensor): Input feature tensor.
        Returns:
            torch.Tensor: Extracted deep features.
        """
        return self.network(x)


class DFKFBlock(nn.Module):
    """
    Implements the Dual-Branch Fuzzy Kinematic Fact (D-FKF) Block.
    """
    def __init__(self, num_features, num_terms, num_rules, hidden_dim):
        """
        Args:
            num_features (int): Number of kinematic features.
            num_terms (int): Number of fuzzy linguistic terms.
            num_rules (int): Number of fuzzy rules.
            hidden_dim (int): Hidden layer dimension for deep features.
        """
        super(DFKFBlock, self).__init__()
        self.fuzzy_inference = FuzzyInferenceModule(num_features, num_terms, num_rules)
        self.deep_feature_extractor = DeepFeatureExtractor(num_features, hidden_dim, num_rules)
        self.feature_fusion = nn.Linear(num_rules * 2, num_rules)  # Combines fuzzy & deep features

    def forward(self, x):
        """
        Processes input kinematic facts and outputs transformed fuzzy kinematic facts.
        Args:
            x (torch.Tensor): Input feature tensor (batch_size, num_features).
        Returns:
            torch.Tensor: Fuzzy kinematic facts (batch_size, num_rules).
        """
        fuzzy_output = self.fuzzy_inference(x)  # (batch_size, num_rules)
        deep_features = self.deep_feature_extractor(x)  # (batch_size, num_rules)
        combined_features = torch.cat((fuzzy_output, deep_features), dim=1)  # Concatenate outputs
        fused_output = F.relu(self.feature_fusion(combined_features))  # Apply fusion layer
        return fused_output



# Example usage
if __name__ == "__main__":
    batch_size = 16
    num_features = 11  # Number of kinematic attributes
    num_terms = 7  # Number of fuzzy linguistic terms
    num_rules = 5  # Number of fuzzy rules
    hidden_dim = 64  # Hidden layer dimension

    dfkf = DFKFBlock(num_features, num_terms, num_rules, hidden_dim)

    # Generate sample input (batch_size, num_features)
    sample_input = torch.randn(batch_size, num_features)
    
    # Forward pass through D-FKF
    fkfs = dfkf(sample_input)
    
    print("Fuzzy Kinematic Facts Output Shape:", fkfs.shape)  # Should be (batch_size, num_rules)
