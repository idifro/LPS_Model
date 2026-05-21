"""CLIP Token Loss: cosine similarity or contrastive (InfoNCE) loss
between predicted tokens and GT text prototypes.

Applied only to Hungarian-matched queries. Unmatched queries are ignored.
Supports linear warmup of the loss weight.

Loss types (config: token.loss_type):
  - 'cosine': 1 - cos_sim(pred, gt_proto)  [default, backward compatible]
  - 'contrastive': InfoNCE with learnable temperature
"""
import torch
from torch import nn
from torch.nn import functional as F


class CLIPTokenLoss(nn.Module):
    """Token loss with configurable type: cosine or contrastive (InfoNCE).

    Args:
        text_prototypes: (num_classes, token_dim) — frozen CLIP text embeddings
        num_classes: number of foreground classes (no-object class excluded)
        loss_type: 'cosine' or 'contrastive'
        init_temperature: initial temperature for contrastive loss (default 0.07)
        learnable_temperature: if True, temperature is a learnable parameter
    """

    def __init__(self, text_prototypes, num_classes=16,
                 loss_type="cosine", init_temperature=0.07,
                 learnable_temperature=True):
        """Initialize CLIPTokenLoss with frozen text prototypes and temperature.

        Registers the CLIP text embeddings as a non-trainable buffer so they
        move with the model to the correct device. For contrastive mode, sets
        up a log-temperature parameter (optionally learnable) that scales the
        cosine similarity logits before cross-entropy, following the CLIP design.

        Args:
            text_prototypes (Tensor): Pre-computed L2-normalized CLIP text
                embeddings of shape (num_classes, token_dim).
            num_classes (int): Number of foreground classes. Queries matched
                to class indices >= num_classes are excluded. Default: 16.
            loss_type (str): Either 'cosine' (1 - cos_sim) or 'contrastive'
                (InfoNCE with temperature scaling). Default: 'cosine'.
            init_temperature (float): Initial temperature tau for contrastive
                loss; stored as log(1/tau). Default: 0.07.
            learnable_temperature (bool): If True and loss_type is 'contrastive',
                the log-temperature is an nn.Parameter optimized during training.
                Default: True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        # Register as buffer (not parameter — frozen, moves with .to(device))
        self.register_buffer("text_prototypes", text_prototypes.float())

        # Learnable log-temperature for contrastive loss (like CLIP)
        if loss_type == "contrastive":
            log_temp = torch.tensor(float(-1.0 * torch.tensor(init_temperature).log()))
            if learnable_temperature:
                self.log_temperature = nn.Parameter(log_temp)
            else:
                self.register_buffer("log_temperature", log_temp)

    def forward(self, pred_tokens, targets, indices):
        """
        Args:
            pred_tokens: (B, Q, token_dim) — L2-normalized predicted tokens
            targets: dict with 'classes' list[Tensor(M,)]
            indices: list of (pred_idx, tgt_idx) tuples from Hungarian matching
        Returns:
            dict with 'loss_token': scalar
        """
        # Gather matched predicted tokens
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])

        if len(src_idx) == 0:
            return {"loss_token": pred_tokens.sum() * 0.0}

        pred_matched = pred_tokens[batch_idx, src_idx]  # (N_matched, token_dim)

        # Gather GT class indices for matched targets
        tgt_classes = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(pred_matched.device)  # (N_matched,)

        # Filter out no-object matches (class >= num_classes)
        valid = tgt_classes < self.num_classes
        if valid.sum() == 0:
            return {"loss_token": pred_tokens.sum() * 0.0}

        pred_valid = pred_matched[valid]  # (N_valid, token_dim)
        gt_classes_valid = tgt_classes[valid]  # (N_valid,)

        if self.loss_type == "contrastive":
            loss = self._contrastive_loss(pred_valid, gt_classes_valid)
        else:
            loss = self._cosine_loss(pred_valid, gt_classes_valid)

        return {"loss_token": loss}

    def _cosine_loss(self, pred_valid, gt_classes_valid):
        """Original cosine similarity loss: 1 - cos_sim(pred, gt_proto)."""
        gt_prototypes = self.text_prototypes[gt_classes_valid]  # (N_valid, token_dim)
        cos_sim = F.cosine_similarity(pred_valid, gt_prototypes, dim=-1)
        return (1.0 - cos_sim).mean()

    def _contrastive_loss(self, pred_valid, gt_classes_valid):
        """InfoNCE contrastive loss with learnable temperature.

        For each matched query, compute cross-entropy over cosine similarities
        to ALL class prototypes. The GT class is the positive; all others are negatives.
        """
        # Cosine similarity: (N_valid, num_classes)
        # pred_valid is already L2-normalized, text_prototypes are L2-normalized
        logits = pred_valid @ self.text_prototypes.t()  # (N_valid, C)

        # Scale by learned temperature: exp(log_temp) = 1/tau
        # Higher log_temp → sharper distribution → stronger gradients
        temperature = self.log_temperature.exp()
        # Clamp temperature to avoid instability (CLIP uses max of 100)
        temperature = temperature.clamp(max=100.0)
        logits = logits * temperature

        # Cross-entropy: gt_classes_valid are the target indices
        loss = F.cross_entropy(logits, gt_classes_valid)
        return loss


def token_warmup_weight(step, warmup_steps):
    """Linear warmup from 0 to 1 over warmup_steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)
