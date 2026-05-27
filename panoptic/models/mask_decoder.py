"""Masked Transformer Decoder for panoptic segmentation.

Adapted from MaskPLS/Mask2Former for single-level backbone features (frozen DisNet 768-d).
Design: 100 thing queries (learnable) + 6 stuff queries (fixed per class) = 106 total.
9 decoder layers (3 blocks x 3 virtual levels, single-level features repeated).
"""
import torch
from torch import nn

from panoptic.models.blocks import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from panoptic.models.positional_encoder import PositionalEncoder
from panoptic.models.clip_token_head import CLIPTokenHead


class PanopticDecoder(nn.Module):
    """Mask2Former-style transformer decoder for panoptic segmentation of 3D point clouds.

    Processes frozen backbone features (e.g., DisNet 768-d) through a stack of
    masked cross-attention, self-attention, and FFN layers to produce per-query
    class logits and binary mask predictions. Designed for single-level features
    with virtual multi-level cycling (num_levels repeats of the same source).

    Pipeline:
        1. Project backbone features to hidden_dim via linear + LayerNorm.
        2. Generate sinusoidal positional encodings from xyz coordinates.
        3. Concatenate learnable thing queries and fixed stuff queries.
        4. Run num_blocks * num_levels decoder layers (cross-attn -> self-attn -> FFN).
        5. At each layer produce intermediate class/mask predictions (deep supervision).
        6. Optionally project queries to CLIP space via CLIPTokenHead.

    Args:
        backbone_dim (int): Dimension of frozen backbone features (default 768).
        hidden_dim (int): Internal transformer hidden dimension (default 256).
        num_classes (int): Number of semantic classes (default 16).
        num_thing_queries (int): Number of learnable instance queries (default 100).
        num_stuff_classes (int): Number of stuff classes with fixed queries (default 6).
        nheads (int): Number of attention heads (default 8).
        dim_feedforward (int): FFN intermediate dimension (default 1024).
        num_blocks (int): Number of repeated decoder blocks (default 3).
        num_levels (int): Virtual feature levels cycled per block (default 3).
        dropout (float): Dropout rate (default 0.0).
        use_token_head (bool): Whether to attach a CLIP token projection head.
        token_dim (int): CLIP embedding dimension (default 768).
    """

    def __init__(
        self,
        backbone_dim=768,
        hidden_dim=256,
        num_classes=16,
        num_thing_queries=100,
        num_stuff_classes=6,
        nheads=8,
        dim_feedforward=1024,
        num_blocks=3,
        num_levels=3,
        dropout=0.0,
        use_token_head=False,
        token_dim=768,
    ):
        """Initialize the panoptic decoder with projections, queries, and layer stacks.

        Builds backbone and mask-feature projection heads, the sinusoidal positional
        encoder, learnable thing/stuff query embeddings, level embeddings for virtual
        multi-scale cycling, and the full stack of cross-attention, self-attention,
        and FFN layers. Also constructs the classification and mask prediction heads,
        plus an optional CLIP token head.

        Args:
            backbone_dim (int): Width of input backbone features.
            hidden_dim (int): Decoder hidden dimension all components share.
            num_classes (int): Semantic class count (output logits have +1 for no-object).
            num_thing_queries (int): Learnable instance query count.
            num_stuff_classes (int): Stuff query count (one per stuff class).
            nheads (int): Multi-head attention head count.
            dim_feedforward (int): FFN expansion dimension.
            num_blocks (int): Number of decoder blocks.
            num_levels (int): Virtual levels per block (total layers = blocks * levels).
            dropout (float): Dropout probability.
            use_token_head (bool): If True, attach CLIPTokenHead for open-vocab.
            token_dim (int): CLIP embedding dimension for token head output.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_thing_queries = num_thing_queries
        self.num_stuff_classes = num_stuff_classes
        self.num_queries = num_thing_queries + num_stuff_classes  # 106
        self.nheads = nheads
        self.num_levels = num_levels
        self.num_layers = num_blocks * num_levels  # 9

        # Backbone adapter: project 768-d → hidden_dim
        self.backbone_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Mask feature projection (same as backbone features for single-level)
        self.mask_feat_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding
        self.pe_layer = PositionalEncoder(feat_size=hidden_dim)

        # Level embeddings (virtual multi-level for cycling)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)

        # Thing queries: learnable
        self.thing_query_feat = nn.Embedding(num_thing_queries, hidden_dim)
        self.thing_query_embed = nn.Embedding(num_thing_queries, hidden_dim)

        # Stuff queries: fixed embeddings per class (learnable but one per stuff class)
        self.stuff_query_feat = nn.Embedding(num_stuff_classes, hidden_dim)
        self.stuff_query_embed = nn.Embedding(num_stuff_classes, hidden_dim)

        # Decoder layers
        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(hidden_dim, nheads, dropout)
            )
            self.self_attn_layers.append(
                SelfAttentionLayer(hidden_dim, nheads, dropout)
            )
            self.ffn_layers.append(
                FFNLayer(hidden_dim, dim_feedforward, dropout)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Optional CLIP token head
        self.token_head = CLIPTokenHead(hidden_dim, token_dim) if use_token_head else None

    def forward(self, backbone_feats, coords, pad_mask=None):
        """
        Args:
            backbone_feats: (B, V, 768) frozen backbone features
            coords: (B, V, 3) voxel xyz coordinates
            pad_mask: (B, V) bool, True where padded (for batched variable-length)
        Returns:
            dict with 'pred_logits' (B, Q, C+1), 'pred_masks' (B, V, Q),
                 'aux_outputs' list of dicts
        """
        B, V, _ = backbone_feats.shape

        # Project backbone features
        src = self.backbone_proj(backbone_feats)  # (B, V, hidden_dim)
        mask_features = self.mask_feat_proj(backbone_feats)  # (B, V, hidden_dim)

        # Add positional encoding to mask features
        pe = self.pe_layer(coords)  # (B, V, hidden_dim)
        mask_features = mask_features + pe

        # Build queries: thing + stuff
        thing_feat = self.thing_query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        thing_pos = self.thing_query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        stuff_feat = self.stuff_query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        stuff_pos = self.stuff_query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        output = torch.cat([thing_feat, stuff_feat], dim=1)  # (B, 106, hidden_dim)
        query_embed = torch.cat([thing_pos, stuff_pos], dim=1)

        predictions_class = []
        predictions_mask = []
        predictions_tokens = []

        # Initial prediction (before any decoder layer)
        outputs_class, outputs_mask, attn_mask = self.pred_heads(
            output, mask_features, pad_mask
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        if self.token_head is not None:
            predictions_tokens.append(self.token_head(self.decoder_norm(output)))

        for i in range(self.num_layers):
            level_idx = i % self.num_levels

            # Add level embedding to source features
            src_level = src + self.level_embed.weight[level_idx].unsqueeze(0).unsqueeze(0)

            # Fix all-masked queries (prevent NaN in attention)
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Cross-attention: queries attend to backbone features
            output = self.cross_attn_layers[i](
                output, src_level,
                attn_mask=attn_mask,
                padding_mask=pad_mask,
                pos=pe,
                query_pos=query_embed,
            )

            # Self-attention: queries attend to each other
            output = self.self_attn_layers[i](
                output, query_pos=query_embed,
            )

            # FFN
            output = self.ffn_layers[i](output)

            # Prediction heads
            outputs_class, outputs_mask, attn_mask = self.pred_heads(
                output, mask_features, pad_mask
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            if self.token_head is not None:
                predictions_tokens.append(self.token_head(self.decoder_norm(output)))

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],  # (B, 106, 17)
            "pred_masks": predictions_mask[-1],     # (B, V, 106)
        }
        if self.token_head is not None:
            out["pred_tokens"] = predictions_tokens[-1]  # (B, 106, token_dim)
        if self.token_head is not None and predictions_tokens:
            out["aux_outputs"] = [
                {"pred_logits": c, "pred_masks": m, "pred_tokens": t}
                for c, m, t in zip(predictions_class[:-1], predictions_mask[:-1],
                                   predictions_tokens[:-1])
            ]
        else:
            out["aux_outputs"] = [
                {"pred_logits": c, "pred_masks": m}
                for c, m in zip(predictions_class[:-1], predictions_mask[:-1])
            ]
        return out

    def pred_heads(self, output, mask_features, pad_mask=None):
        """Compute class logits, mask logits, and attention mask for next layer."""
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)       # (B, Q, C+1)
        mask_embed = self.mask_embed(decoder_output)           # (B, Q, hidden_dim)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)  # (B, V, Q)

        # Attention mask for next cross-attention: mask out points where sigmoid < 0.5
        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()  # (B, V, Q)
        if pad_mask is not None:
            attn_mask[pad_mask.unsqueeze(-1).expand_as(attn_mask)] = True

        # Reshape for multi-head attention: (B*nheads, Q, V)
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask
