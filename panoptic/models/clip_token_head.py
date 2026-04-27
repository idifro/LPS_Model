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
