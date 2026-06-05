"""Enhanced token loss with 216+ negative bank (L5/C1).

Replaces the 16-class InfoNCE in CLIPTokenLoss with a 216+ denominator
to prevent closed-set collapse. Includes multi-prompt augmentation (T4).

Usage:
    from panoptic.losses.token_loss_c1 import TokenLossC1

    loss_fn = TokenLossC1(
        base_protos_path="panoptic/clip_prototypes/text_prototypes_vitl14_224.pt",
        negative_bank_path="panoptic/clip_prototypes/negative_bank_200.pt",
        multiprompt_path="panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt",
    )
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLossC1(nn.Module):
    """InfoNCE token loss with 216+ negative denominator.

    The old 16-class denominator caused the token head to learn 16 narrow tunnels
    in CLIP space, making novel-class detection impossible. With 216+ denominators,
    the head must learn fine-grained CLIP-space structure.

    Args:
        base_protos_path: path to base 16-class prototypes .pt file
        negative_bank_path: path to negative bank .pt file (200, 768)
        multiprompt_path: optional path to multi-prompt prototypes (16, 10, 768)
        learnable_temperature: whether temperature is learnable
        initial_temperature: initial temperature value (0.07 = CLIP default)
    """

    def __init__(
        self,
        base_protos_path: str,
        negative_bank_path: str,
        multiprompt_path: str = None,
        learnable_temperature: bool = True,
        initial_temperature: float = 0.07,
        temp_max: float = 5.0,
        margin: float = 0.1,
        novel_ids: list = None,
    ):
        super().__init__()

        self.margin = margin
        self.temp_max = temp_max

        # Load base prototypes
        base_data = torch.load(base_protos_path, map_location="cpu")
        if isinstance(base_data, dict):
            base_protos = base_data["embeddings"]  # (16, 768)
        else:
            base_protos = base_data
        base_protos = F.normalize(base_protos.float(), dim=-1)

        # B12/N4: separate base training IDs from novel IDs
        # novel_ids are included as soft negatives (in denominator only, never as targets)
        if novel_ids is None:
            novel_ids = [2, 5, 6, 15]  # bus, motorcycle, pedestrian, vegetation
        self.novel_ids = set(novel_ids)
        self.base_ids = [i for i in range(base_protos.shape[0]) if i not in self.novel_ids]
        self.n_base_train = len(self.base_ids)  # 12 (classes used as targets)
        self.n_base = base_protos.shape[0]  # 16 (all protos including novel)

        # Load negative bank
        neg_data = torch.load(negative_bank_path, map_location="cpu")
        if isinstance(neg_data, dict):
            neg_protos = neg_data["embeddings"]  # (200, 768)
        else:
            neg_protos = neg_data
        neg_protos = F.normalize(neg_protos.float(), dim=-1)

        # Concatenate: [16 base (incl. novel as soft neg) ; 200 negatives] = 216 total
        # Novel protos sit at their original indices (2,6,14,15) — never used as targets
        all_protos = torch.cat([base_protos, neg_protos], dim=0)  # (216, 768)
        self.register_buffer("all_protos", all_protos)
        self.n_total = all_protos.shape[0]  # 216

        # Multi-prompt prototypes (optional, for T4)
        if multiprompt_path is not None:
            mp_data = torch.load(multiprompt_path, map_location="cpu")
            if isinstance(mp_data, dict):
                mp_protos = mp_data["embeddings"]  # (16, 10, 768)
            else:
                mp_protos = mp_data
            mp_protos = F.normalize(mp_protos.float(), dim=-1)
            self.register_buffer("multiprompt_protos", mp_protos)
            self.n_templates = mp_protos.shape[1]
        else:
            self.multiprompt_protos = None
            self.n_templates = 0

        # Temperature — initialize at log(3.0) ≈ 1.1 so it starts BELOW temp_max
        # and can learn freely. Old init log(1/0.07)=2.66 → exp=14.3 → clamped to max,
        # which zeroed all gradients (stuck at ceiling).
        log_temp = torch.tensor(np.log(3.0))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=1.0, max=self.temp_max)

    def _get_protos_for_step(self):
        """Get prototype matrix for this step (with optional T4 multi-prompt).

        If multiprompt is enabled, randomly samples one template per class
        and replaces the base prototypes with that template's embedding.

        Returns:
            protos: (N_total, 768) where first n_base entries may vary per step
        """
        if self.multiprompt_protos is not None and self.training:
            # Random template index per class
            template_idx = torch.randint(
                0, self.n_templates, (self.n_base,),
                device=self.all_protos.device
            )
            # Gather: (16, 768) from (16, 10, 768)
            sampled_base = self.multiprompt_protos[
                torch.arange(self.n_base, device=self.all_protos.device),
                template_idx
            ]
            # Replace base protos with sampled templates
            protos = self.all_protos.clone()
            protos[:self.n_base] = sampled_base
            return protos
        return self.all_protos

    def forward(self, pred_embeds, targets, indices):
        """Compute C1 token loss on matched queries.

        Args:
            pred_embeds: (B, Q, 768) L2-normalized query embeddings
            targets: list of dicts with 'labels' (M,) per batch item
            indices: list of (pred_idx, gt_idx) tuples from Hungarian matching

        Returns:
            loss: scalar tensor
        """
        device = pred_embeds.device
        protos = self._get_protos_for_step()  # (216, 768)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Get matched query embeddings
            matched_embeds = pred_embeds[b, pred_idx]  # (M, 768)

            # Get GT class labels for matched queries
            # Support both dict-of-lists format {"classes": [tensor, ...]}
            # and list-of-dicts format [{"labels": tensor}, ...]
            if isinstance(targets, dict) and "classes" in targets:
                gt_classes = targets["classes"][b][gt_idx]
            else:
                gt_classes = targets[b]["labels"][gt_idx]  # (M,)

            # Filter out no-object class AND novel classes (not supervised)
            valid = torch.tensor(
                [(c.item() < self.n_base and c.item() not in self.novel_ids)
                 for c in gt_classes],
                dtype=torch.bool, device=device
            )
            if valid.sum() == 0:
                continue

            matched_embeds = matched_embeds[valid]  # (M', 768)
            gt_classes = gt_classes[valid]  # (M',) indices from base_ids only

            # Cosine similarities to all 216 concepts
            sim = matched_embeds @ protos.T  # (M', 216)

            # Margin penalty: subtract margin from GT class similarity
            # This forces the model to achieve margin > self.margin for correct classification
            sim_margin = sim.clone()
            sim_margin[torch.arange(len(gt_classes), device=device), gt_classes] -= self.margin

            sim_margin = sim_margin / self.temperature  # scale by learned temperature

            # Cross-entropy: GT labels index into their original positions (0-15)
            loss_b = F.cross_entropy(sim_margin, gt_classes)

            if torch.isfinite(loss_b):
                total_loss = total_loss + loss_b
                n_valid += 1

        if n_valid == 0:
            return total_loss

        return total_loss / n_valid
