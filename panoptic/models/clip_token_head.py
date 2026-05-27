"""CLIP Token Head: projects decoder query embeddings into CLIP text embedding space.

Architecture: MLP(hidden_dim → hidden_dim*2 → token_dim) + L2 normalization.
Attaches to decoder_output (same tensor used by class_embed and mask_embed).
"""
import torch
from torch import nn
from torch.nn import functional as F

from panoptic.models.blocks import MLP


class CLIPTokenHead(nn.Module):
    """Predicts L2-normalized CLIP-space token embeddings per query."""

    def __init__(self, hidden_dim=256, token_dim=768, num_layers=3):
        """Initialize the CLIP token projection head.

        Builds a multi-layer perceptron that maps decoder query embeddings
        (hidden_dim) to the CLIP text embedding space (token_dim). The MLP
        uses an intermediate width of hidden_dim*2. The forward pass applies
        L2 normalization after the MLP so outputs lie on the unit hypersphere.

        Args:
            hidden_dim (int): Input dimension matching decoder hidden size.
            token_dim (int): Output dimension matching CLIP text embedding size.
            num_layers (int): Number of linear layers in the projection MLP.
        """
        super().__init__()
        self.mlp = MLP(hidden_dim, hidden_dim * 2, token_dim, num_layers)

    def forward(self, decoder_output):
        """
        Args:
            decoder_output: (B, Q, hidden_dim) — normalized decoder query embeddings
        Returns:
            tokens: (B, Q, token_dim) — L2-normalized CLIP-space embeddings
        """
        tokens = self.mlp(decoder_output)  # (B, Q, token_dim)
        tokens = F.normalize(tokens, dim=-1)  # L2 normalize
        return tokens
