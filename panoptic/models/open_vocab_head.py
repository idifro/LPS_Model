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


class RegionCLIPHead(nn.Module):
    """Projects decoder query embeddings into CLIP feature space.
    
    Unlike CLIPTokenHead which just projects queries, this also supports
    mask-weighted pooling of backbone features for region-level CLIP alignment.
    
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
        return F.normalize(out, dim=-1)


class ObjectnessHead(nn.Module):
    """Predicts objectness scores: thing / stuff / no-object.
    
    This is class-AGNOSTIC — it doesn't predict which class, just whether
    a query corresponds to a thing instance, stuff region, or nothing.
    """
    
    def __init__(self, hidden_dim=256, mode="ternary"):
        """
        Args:
            hidden_dim: Input dimension from decoder queries
            mode: "ternary" (thing/stuff/no-obj) or "binary" (object/no-obj)
        """
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
    """Pools backbone features per predicted mask to get region-level CLIP features.
    
    This is KEY for open-vocab: the backbone already outputs CLIP-aligned 768-d features.
    By pooling them per mask, we get a region-level CLIP descriptor that can be compared
    against ANY text query at inference time.
    """
    
    def __init__(self, mode="weighted_mean"):
        """
        Args:
            mode: "mean" (simple average over mask voxels) or 
                  "weighted_mean" (mask-probability-weighted average)
        """
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
        Q = pred_masks.shape[-1]
        
        # Softly weight features by mask probability
        mask_probs = pred_masks.sigmoid()  # (B, V, Q)
        
        if self.mode == "weighted_mean":
            # Weighted average: sum(mask_prob * feat) / sum(mask_prob)
            # (B, V, Q) -> transpose -> (B, Q, V) @ (B, V, D) -> (B, Q, D)
            weights = mask_probs.permute(0, 2, 1)  # (B, Q, V)
            weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, Q, 1)
            pooled = torch.bmm(weights, backbone_feats) / weight_sum  # (B, Q, D)
        else:
            # Hard mask pooling (threshold at 0.5)
            hard_masks = (mask_probs > 0.5).float()  # (B, V, Q)
            weights = hard_masks.permute(0, 2, 1)  # (B, Q, V)
            weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            pooled = torch.bmm(weights, backbone_feats) / weight_sum
        
        # L2-normalize to CLIP space
        pooled = F.normalize(pooled, dim=-1)
        return pooled


class OpenVocabPanopticHead(nn.Module):
    """Combined open-vocabulary panoptic prediction head.
    
    Replaces the closed-set (class_embed + token_head) with:
    - objectness_head: thing/stuff/no-object
    - region_clip_head: query → CLIP space (for text-based classification)
    - region_pooler: backbone features → per-mask CLIP descriptor
    
    At inference, classification = cosine_sim(region_feature, text_embedding)
    weighted by objectness confidence.
    """
    
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
