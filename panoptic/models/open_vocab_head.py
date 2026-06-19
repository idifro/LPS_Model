"""Open-vocabulary region-CLIP alignment head.

This replaces the closed-set 16-class linear head with:
1. Objectness head: predicts thing/stuff/no-object (class-agnostic)
2. Region-CLIP head: projects query embeddings to CLIP space for open-vocab matching
3. Region feature pooling: aggregates backbone CLIP features per predicted mask

At inference:
  score = objectness_score * cosine_sim(region_clip_feature, text_prompt_embedding)

This enables queries like "a red SUV" or "car near black building" at test time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_normalize(x, dim=-1, eps=1e-6):
    """L2-normalize with a safe epsilon to avoid NaN on zero vectors."""
    norm = x.norm(dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


class RegionCLIPHead(nn.Module):
    """Projects decoder query embeddings into CLIP feature space.

    Architecture:
        query (256-d) → MLP → L2-normalized CLIP embedding (768-d)
    """

    def __init__(self, hidden_dim=256, clip_dim=768, num_layers=3):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, clip_dim))
        self.mlp = nn.Sequential(*layers)
        self.clip_dim = clip_dim

    def forward(self, query_features):
        """
        Args:
            query_features: (B, Q, hidden_dim) decoder query outputs
        Returns:
            (B, Q, clip_dim) L2-normalized CLIP-space embeddings
        """
        out = self.mlp(query_features)
        return safe_normalize(out, dim=-1)


class ObjectnessHead(nn.Module):
    """Predicts objectness scores: thing / stuff / no-object (class-agnostic)."""

    def __init__(self, hidden_dim=256, mode="ternary"):
        super().__init__()
        self.mode = mode
        num_classes = 3 if mode == "ternary" else 2
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, query_features):
        """
        Args:
            query_features: (B, Q, hidden_dim)
        Returns:
            (B, Q, num_classes) logits
        """
        return self.head(query_features)


class RegionPooler(nn.Module):
    """Pools backbone features per predicted mask to get region-level CLIP features."""

    def __init__(self, mode="weighted_mean"):
        super().__init__()
        self.mode = mode

    def forward(self, backbone_feats, pred_masks):
        """
        Args:
            backbone_feats: (B, V, D) raw backbone features (768-d CLIP space)
            pred_masks: (B, V, Q) mask logits from decoder
        Returns:
            (B, Q, D) pooled region features in CLIP space
        """
        B, V, D = backbone_feats.shape

        mask_probs = pred_masks.sigmoid()  # (B, V, Q)

        if self.mode == "weighted_mean":
            weights = mask_probs.permute(0, 2, 1)          # (B, Q, V)
            weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-3)  # (B, Q, 1)
            pooled = torch.bmm(weights, backbone_feats) / weight_sum        # (B, Q, D)
        else:
            hard_masks = (mask_probs > 0.5).float()
            weights = hard_masks.permute(0, 2, 1)
            weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-3)
            pooled = torch.bmm(weights, backbone_feats) / weight_sum

        # Safe L2-normalize — avoids NaN when pooled vector has near-zero norm
        # (happens when all mask probs are ~0 for a query with no attended voxels)
        pooled = safe_normalize(pooled, dim=-1)
        return pooled


class OpenVocabPanopticHead(nn.Module):
    """Combined open-vocabulary panoptic prediction head."""

    def __init__(self, hidden_dim=256, clip_dim=768, num_mlp_layers=3,
                 objectness_mode="ternary", pool_mode="weighted_mean"):
        super().__init__()
        self.objectness_head = ObjectnessHead(hidden_dim, objectness_mode)
        self.region_clip_head = RegionCLIPHead(hidden_dim, clip_dim, num_mlp_layers)
        self.region_pooler = RegionPooler(pool_mode)
        self.clip_dim = clip_dim

        # Learnable temperature for contrastive loss
        self.log_temperature = nn.Parameter(torch.tensor(2.6592))  # ~ln(14.3)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, query_features, backbone_feats=None, pred_masks=None):
        """
        Args:
            query_features: (B, Q, hidden_dim) from decoder
            backbone_feats: (B, V, 768) raw backbone CLIP features (optional)
            pred_masks: (B, V, Q) mask logits (optional, for region pooling)
        Returns:
            dict with:
                objectness: (B, Q, 2or3) objectness logits
                query_clip: (B, Q, 768) projected query CLIP features
                region_clip: (B, Q, 768) pooled backbone CLIP features (if available)
        """
        out = {
            "objectness": self.objectness_head(query_features),
            "query_clip": self.region_clip_head(query_features),
        }

        if backbone_feats is not None and pred_masks is not None:
            out["region_clip"] = self.region_pooler(backbone_feats, pred_masks)

        return out