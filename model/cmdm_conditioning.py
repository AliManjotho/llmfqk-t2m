import torch
import torch.nn as nn
from transformers import BertModel
from dfkf import DFKFBlock
from llm_ar import LLMAmbiguityResolver

class C_MDM_Conditioning(nn.Module):
    """
    Implements contextual conditioning for the Motion Diffusion Model (C-MDM).
    """

    def __init__(self, text_embedding_dim=768, fkf_dim=256, context_dim=512, hidden_dim=512):
        """
        Args:
            text_embedding_dim (int): Dimensionality of text embeddings from BERT/LLM.
            fkf_dim (int): Dimensionality of extracted FKFs from D-FKF block.
            context_dim (int): Dimensionality of contextual embeddings from LLM-AR.
            hidden_dim (int): Hidden layer dimension for processing the conditioning signal.
        """
        super(C_MDM_Conditioning, self).__init__()

        # Load pretrained BERT model for text embedding
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Fuzzy Kinematic Facts (FKF) encoder
        self.fkf_encoder = nn.Sequential(
            nn.Linear(fkf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Contextual Term Encoder (LLM-AR output)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Final conditioning projection layer
        self.conditioning_projector = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, text_input, fkf_input, context_input):
        """
        Computes the conditioning signal for C-MDM.
        Args:
            text_input (str): Input textual description.
            fkf_input (torch.Tensor): FKF features from D-FKF block (batch_size, fkf_dim).
            context_input (torch.Tensor): Contextual embeddings from LLM-AR (batch_size, context_dim).
        Returns:
            torch.Tensor: Combined conditioning representation (batch_size, hidden_dim).
        """
        # Encode text input using BERT
        inputs = self.text_encoder(text_input, return_tensors="pt", truncation=True, padding=True)
        text_features = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)  # Extract sentence embedding

        # Encode FKFs
        fkf_features = self.fkf_encoder(fkf_input)

        # Encode Contextual Terms
        context_features = self.context_encoder(context_input)

        # Concatenate all context features
        conditioning_features = torch.cat([text_features, fkf_features, context_features], dim=-1)

        # Project to final conditioning representation
        final_conditioning = self.conditioning_projector(conditioning_features)

        return final_conditioning



if __name__ == "__main__":
    batch_size = 16
    text_input = ["A person is walking slowly"] * batch_size
    fkf_dim = 256
    context_dim = 512
    hidden_dim = 512

    # Sample FKFs from D-FKF
    fkf_input = torch.randn(batch_size, fkf_dim)

    # Sample contextual embeddings from LLM-AR
    context_input = torch.randn(batch_size, context_dim)

    # Initialize C-MDM Conditioning Module
    cmdm_conditioning = C_MDM_Conditioning(text_embedding_dim=768, fkf_dim=fkf_dim, context_dim=context_dim, hidden_dim=hidden_dim)

    # Generate conditioning signal
    conditioning_signal = cmdm_conditioning(text_input, fkf_input, context_input)
    
    print("Conditioning Signal Shape:", conditioning_signal.shape)  # Should be (batch_size, hidden_dim)
