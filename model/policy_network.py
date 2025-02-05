import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

class PolicyNetwork(nn.Module):
    """
    Implements a reinforcement learning-based policy network for selecting in-context examples.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Input feature size (BERT embedding size).
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Number of contextual examples to select (K).
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass through the policy network.
        Args:
            x (torch.Tensor): Input embeddings from BERT (batch_size, input_dim).
        Returns:
            torch.Tensor: Probability distribution over in-context examples (batch_size, output_dim).
        """
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


class RLBasedFSICL:
    """
    Reinforcement learning-based Few-Shot In-Context Learning (FS-ICL) for selecting optimal examples.
    """

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=10, lr=1e-4, gamma=0.99):
        """
        Args:
            input_dim (int): BERT embedding size.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Number of in-context examples.
            lr (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []

    def select_examples(self, text_embedding):
        """
        Selects in-context examples using the policy network.
        Args:
            text_embedding (torch.Tensor): Embedding from BERT (batch_size, input_dim).
        Returns:
            torch.Tensor: Selected examples' indices.
        """
        probs = self.policy_net(text_embedding)
        distribution = torch.distributions.Categorical(probs)
        selected_indices = distribution.sample()
        self.log_probs.append(distribution.log_prob(selected_indices))
        return selected_indices

    def compute_rewards(self, motion_quality, text_alignment, context_coherence):
        """
        Computes rewards based on motion quality, text-motion alignment, and contextual coherence.
        Args:
            motion_quality (float): Quality of generated motion.
            text_alignment (float): Cosine similarity of text and motion embeddings.
            context_coherence (float): Cosine similarity of contextual embeddings.
        Returns:
            float: Computed reward.
        """
        lambda_q, lambda_a, lambda_c = 0.02, 0.05, 0.12
        reward = lambda_q * motion_quality + lambda_a * text_alignment + lambda_c * context_coherence
        return reward

    def update_policy(self):
        """
        Updates policy network using rewards.
        """
        R = 0
        policy_loss = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            policy_loss.append(-R * self.log_probs.pop())

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.rewards = []




if __name__ == "__main__":
    batch_size = 16
    input_dim = 768  # BERT embedding size
    policy = RLBasedFSICL(input_dim=input_dim)

    # Example BERT text embeddings (random tensor for simulation)
    text_embeddings = torch.randn(batch_size, input_dim)

    # Select in-context examples
    selected_examples = policy.select_examples(text_embeddings)
    print("Selected Examples:", selected_examples)

    # Simulated reward values
    motion_quality = 0.9
    text_alignment = 0.85
    context_coherence = 0.88

    # Compute and store reward
    reward = policy.compute_rewards(motion_quality, text_alignment, context_coherence)
    policy.rewards.append(reward)

    # Update policy network
    policy.update_policy()