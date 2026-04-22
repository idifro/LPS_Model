"""Transformer building blocks for panoptic decoder.

Adapted from MaskPLS (Marcuzzi et al.) / Mask2Former (Facebook).
Compatible with PyTorch 1.7.1 (no batch_first — manual transpose).
"""
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
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
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
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
    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
