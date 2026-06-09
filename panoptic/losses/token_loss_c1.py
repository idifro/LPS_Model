"""Enhanced token loss with 216+ negative bank (L5/C1).

Replaces the 16-class InfoNCE in CLIPTokenLoss with a 216+ denominator
to prevent closed-set collapse. Includes multi-prompt augmentation (T4).

Usage:
    from panoptic.losses.token_loss_c1 import TokenLossC1

    loss_fn = TokenLossC1(
        base_protos_path="panoptic/clip_prototypes/text_prototypes_vitl14_224.pt",
        negative_bank_path="panoptic/clip_prototypes/negative_bank_200.pt",
        multiprompt_path="panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt.pt",
        exclude_novel_from_denom=True,   # recommended: remove novel protos from denominator
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
        exclude_novel_from_denom: if True, novel class protos are removed from the
            InfoNCE denominator entirely. This prevents the token loss from actively
            pushing embeddings away from novel prototypes (the primary cause of
            novel PQ=0 in the B12/N4 split).
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
        exclude_novel_from_denom: bool = True,
    ):
        super().__init__()

        self.margin = margin
        self.temp_max = temp_max
        self.exclude_novel_from_denom = exclude_novel_from_denom

        # Load base prototypes
        base_data = torch.load(base_protos_path, map_location="cpu")
        if isinstance(base_data, dict):
            base_protos = base_data["embeddings"]  # (16, 768)
        else:
            base_protos = base_data
        base_protos = F.normalize(base_protos.float(), dim=-1)

        # B12/N4: separate base training IDs from novel IDs
        if novel_ids is None:
            novel_ids = [2, 5, 6, 15]  # bus, motorcycle, pedestrian, vegetation
        self.novel_ids = set(novel_ids)
        self.base_ids = [i for i in range(base_protos.shape[0]) if i not in self.novel_ids]
        self.n_base_train = len(self.base_ids)  # 12
        self.n_base = base_protos.shape[0]      # 16

        # Load negative bank
        neg_data = torch.load(negative_bank_path, map_location="cpu")
        if isinstance(neg_data, dict):
            neg_protos = neg_data["embeddings"]  # (200, 768)
        else:
            neg_protos = neg_data
        neg_protos = F.normalize(neg_protos.float(), dim=-1)

        if exclude_novel_from_denom:
            # Build denominator from base-only protos + negative bank (212 total).
            # Novel protos are absent — the token loss cannot push embeddings away from them.
            base_only = base_protos[self.base_ids]          # (12, 768)
            all_protos = torch.cat([base_only, neg_protos], dim=0)   # (212, 768)
            # Remap: original class index (0-15) → compact index (0-11) in all_protos
            self.label_remap = {orig: i for i, orig in enumerate(self.base_ids)}
        else:
            # Legacy: novel protos included as soft negatives (never targets)
            all_protos = torch.cat([base_protos, neg_protos], dim=0)  # (216, 768)
            self.label_remap = {i: i for i in self.base_ids}

        self.register_buffer("all_protos", all_protos)
        self.n_total = all_protos.shape[0]

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

        # Temperature — initialize at log(0.5) so it starts well below temp_max
        # and can decay toward the 0.07 floor (CLIP default).
        # log(3.0) was the old init; it wasted early epochs with near-zero gradient.
        log_temp = torch.tensor(np.log(0.5))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.07, max=self.temp_max)

    def _get_protos_for_step(self):
        """Get prototype matrix for this step (with optional T4 multi-prompt).

        Samples one template per BASE class and replaces the leading entries
        of all_protos (which are always base-only regardless of exclude_novel_from_denom).

        Returns:
            protos: (N_total, 768)
        """
        if self.multiprompt_protos is not None and self.training:
            template_idx = torch.randint(
                0, self.n_templates, (self.n_base_train,),
                device=self.all_protos.device
            )
            # Gather only base class rows from multiprompt_protos (16, 10, 768)
            base_ids_t = torch.tensor(self.base_ids, device=self.all_protos.device)
            sampled_base = self.multiprompt_protos[base_ids_t, template_idx]  # (12, 768)
            protos = self.all_protos.clone()
            protos[:self.n_base_train] = sampled_base
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
        protos = self._get_protos_for_step()  # (212 or 216, 768)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            matched_embeds = pred_embeds[b, pred_idx]  # (M, 768)

            if isinstance(targets, dict) and "classes" in targets:
                gt_classes = targets["classes"][b][gt_idx]
            else:
                gt_classes = targets[b]["labels"][gt_idx]  # (M,)

            # Filter: keep only base-class matches (novel classes are not supervised)
            valid = torch.tensor(
                [(c.item() < self.n_base and c.item() not in self.novel_ids)
                 for c in gt_classes],
                dtype=torch.bool, device=device
            )
            if valid.sum() == 0:
                continue

            matched_embeds = matched_embeds[valid]       # (M', 768)
            gt_classes     = gt_classes[valid]           # (M',) original 0-15 indices

            # Remap GT labels to compact indices into all_protos
            gt_remapped = torch.tensor(
                [self.label_remap[c.item()] for c in gt_classes],
                dtype=torch.long, device=device
            )                                            # (M',) indices 0-11

            sim = matched_embeds @ protos.T              # (M', 212 or 216)

            sim_margin = sim.clone()
            sim_margin[torch.arange(len(gt_remapped), device=device), gt_remapped] -= self.margin
            sim_margin = sim_margin / self.temperature

            loss_b = F.cross_entropy(sim_margin, gt_remapped)

            if torch.isfinite(loss_b):
                total_loss = total_loss + loss_b
                n_valid += 1

        if n_valid == 0:
            return total_loss

        return total_loss / n_valid
