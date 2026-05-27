"""Transformer building blocks for panoptic decoder.

Adapted from MaskPLS (Marcuzzi et al.) / Mask2Former (Facebook).
Compatible with PyTorch 1.7.1 (no batch_first — manual transpose).
"""
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    """Pre-norm multi-head self-attention with residual connection.

    Used inside the panoptic decoder so that object queries can exchange
    information among themselves after cross-attending to backbone features.
    Applies positional embeddings to queries/keys, runs MHA, adds residual,
    then layer-normalizes.

    Args:
        d_model (int): Hidden dimension of queries.
        nhead (int): Number of attention heads.
        dropout (float): Dropout probability on attention weights and residual.
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        """Initialize self-attention layer with MHA, LayerNorm, and dropout.

        Constructs the PyTorch MultiheadAttention module, a LayerNorm for
        post-residual normalization, and a dropout layer. Xavier-uniform
        initialization is applied to all weight matrices via _reset_parameters.

        Args:
            d_model (int): Model hidden dimension, used for MHA embed_dim and LayerNorm.
            nhead (int): Number of parallel attention heads.
            dropout (float): Dropout rate for attention and residual paths.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier uniform initialization to all multi-dimensional parameters.

        Iterates over every parameter in the layer and applies Xavier uniform
        init to weight matrices (dim > 1), leaving biases (dim == 1) at their
        default initialization. Called automatically at the end of __init__.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embedding to a tensor if the embedding is provided.

        Utility used to inject query/key positional information before attention.
        If pos is None the tensor passes through unchanged, enabling optional
        positional encoding without branching logic in the forward pass.

        Args:
            tensor (Tensor): Input tensor of shape (B, N, C).
            pos (Optional[Tensor]): Positional embedding of same shape, or None.

        Returns:
            Tensor: tensor + pos, or tensor unchanged if pos is None.
        """
        return tensor if pos is None else tensor + pos

    def forward(self, q_embed, attn_mask=None, padding_mask=None, query_pos=None):
        """q_embed: (B, N, C) batch-first input."""
        q = k = self.with_pos_embed(q_embed, query_pos)
        # MHA expects (N, B, C) — sequence first
        q_embed2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), value=q_embed.transpose(0, 1),
            attn_mask=attn_mask, key_padding_mask=padding_mask,
        )[0].transpose(0, 1)
        q_embed = q_embed + self.dropout(q_embed2)
        q_embed = self.norm(q_embed)
        return q_embed


class CrossAttentionLayer(nn.Module):
    """Pre-norm multi-head cross-attention where queries attend to backbone features.

    In each decoder layer the object queries cross-attend to the projected
    backbone point features. The layer first normalizes queries, adds positional
    embeddings to both queries and keys/values, runs MHA, then adds the
    residual. Mask-based attention (predicted mask logits thresholded at 0.5)
    is supported via attn_mask.

    Args:
        d_model (int): Hidden dimension of queries and keys.
        nhead (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        """Initialize cross-attention layer with MHA, LayerNorm, and dropout.

        Constructs nn.MultiheadAttention for cross-attention, a LayerNorm
        applied pre-attention to the query, and a dropout layer for the
        residual path. Weights are Xavier-uniform initialized.

        Args:
            d_model (int): Embedding dimension for queries and keys/values.
            nhead (int): Number of parallel attention heads.
            dropout (float): Dropout rate for attention and residual.
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier uniform initialization to all multi-dimensional parameters.

        Iterates over every parameter in the layer and applies Xavier uniform
        init to weight matrices (dim > 1), leaving biases at their default.
        Called automatically at the end of __init__.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embedding to a tensor if the embedding is provided.

        Enables optional positional encoding injection for queries or keys
        without branching. Returns tensor unchanged when pos is None.

        Args:
            tensor (Tensor): Input of shape (B, N, C).
            pos (Optional[Tensor]): Positional encoding of same shape, or None.

        Returns:
            Tensor: tensor + pos, or tensor if pos is None.
        """
        return tensor if pos is None else tensor + pos

    def forward(self, q_embed, bb_feat, attn_mask=None, padding_mask=None,
                pos=None, query_pos=None):
        """q_embed: (B, Q, C), bb_feat: (B, V, C) — batch-first inputs."""
        q_embed = self.norm(q_embed)
        query = self.with_pos_embed(q_embed, query_pos).transpose(0, 1)
        key = self.with_pos_embed(bb_feat, pos).transpose(0, 1)
        value = self.with_pos_embed(bb_feat, pos).transpose(0, 1)
        q_embed2 = self.multihead_attn(
            query=query, key=key, value=value,
            attn_mask=attn_mask, key_padding_mask=padding_mask,
        )[0].transpose(0, 1)
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


class FFNLayer(nn.Module):
    """Pre-norm position-wise feedforward network with residual connection.

    Applied after self-attention in each decoder layer. First normalizes the
    input, then passes it through a two-layer MLP (d_model -> dim_feedforward
    -> d_model) with ReLU activation, and adds the residual. This expands
    representational capacity between attention operations.

    Args:
        d_model (int): Input and output dimension.
        dim_feedforward (int): Hidden dimension of the inner linear layer.
        dropout (float): Dropout probability after ReLU and on residual.
    """

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0):
        """Initialize FFN with two linear layers, LayerNorm, dropout, and ReLU.

        Constructs Linear(d_model, dim_feedforward), Linear(dim_feedforward, d_model),
        a LayerNorm for pre-norm, and dropout. All weight matrices are Xavier-uniform
        initialized via _reset_parameters.

        Args:
            d_model (int): Model hidden dimension (input and output width).
            dim_feedforward (int): Width of the intermediate expansion layer.
            dropout (float): Dropout probability applied after activation and on residual.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier uniform initialization to all multi-dimensional parameters.

        Iterates over all parameters and applies Xavier uniform to matrices
        (dim > 1). Biases are left at default initialization. Called at the
        end of __init__.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        """Apply pre-norm FFN with residual to the input tensor.

        Steps: LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> Dropout
        -> add residual. The input and output shapes are identical.

        Args:
            tgt (Tensor): Input of shape (..., d_model).

        Returns:
            Tensor: Output of same shape with FFN transformation applied.
        """
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize a multi-layer perceptron with ReLU activations.

        Builds a sequence of Linear layers: the first maps input_dim to
        hidden_dim, intermediate layers map hidden_dim to hidden_dim, and the
        final layer maps hidden_dim to output_dim. ReLU is applied between
        all layers except after the last.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Width of all intermediate hidden layers.
            output_dim (int): Dimensionality of the output features.
            num_layers (int): Total number of linear layers (must be >= 1).
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Pass input through all linear layers with ReLU on intermediates.

        Applies each linear layer sequentially. ReLU activation is applied
        after every layer except the last, which produces raw logits or
        embeddings depending on downstream usage.

        Args:
            x (Tensor): Input tensor of shape (..., input_dim).

        Returns:
            Tensor: Output of shape (..., output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
