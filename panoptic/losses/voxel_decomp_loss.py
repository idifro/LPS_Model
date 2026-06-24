"""L7: Voxel decomposition loss — per-voxel InfoNCE on backbone features.

Trains LoRA adapters on the frozen backbone to make backbone features more
discriminative in CLIP text space. Base classes only (NOVEL_IDS excluded from
positives and denominator), class-balanced voxel sampling.

Loss:  InfoNCE = CE(logits, class_id)
  where logits[i] = concat(protos, neg_bank) @ feat[i] / temperature
  positive = proto[cls_id], negatives = all other protos + full neg bank

Unlike L_O / L_V which operate through the decoder, L7 gradients flow directly
through backbone LoRA adapters, providing a complementary training signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_NOVEL_IDS = frozenset({2, 5, 6, 15})  # bus, motorcycle, pedestrian, vegetation (0-indexed)
_IGNORE_LABEL = 255


class VoxelDecompLoss(nn.Module):

    def __init__(self, proto_path, neg_bank_path, temperature=0.07, n_sample=8192):
        """
        Args:
            proto_path:    path to text_prototypes .pt (dict with "embeddings" or plain tensor)
            neg_bank_path: path to negative_bank .pt
            temperature:   InfoNCE temperature (default 0.07)
            n_sample:      max total voxels to sample per call (class-balanced)
        """
        super().__init__()
        self.temperature = temperature
        self.n_sample = n_sample

        proto_data = torch.load(proto_path, map_location="cpu")
        protos = proto_data["embeddings"] if isinstance(proto_data, dict) else proto_data
        protos = F.normalize(protos.float(), dim=-1)   # (16, 768)
        self.register_buffer("protos", protos)

        neg_data = torch.load(neg_bank_path, map_location="cpu")
        if isinstance(neg_data, dict):
            neg_bank = neg_data.get("embeddings",
                       neg_data.get("neg_bank",
                       next(iter(neg_data.values()))))
        else:
            neg_bank = neg_data
        neg_bank = F.normalize(neg_bank.float(), dim=-1)   # (M, 768)
        self.register_buffer("neg_bank", neg_bank)

        self.base_ids = [i for i in range(16) if i not in _NOVEL_IDS]

    def forward(self, backbone_feats, vox_sem):
        """
        Args:
            backbone_feats: (V, 768) raw backbone features (requires_grad via LoRA)
            vox_sem:        (V,)     0-indexed semantic label (255 = ignore)

        Returns:
            loss: scalar tensor (0 if no base-class voxels found)
        """
        device = backbone_feats.device
        feats  = F.normalize(backbone_feats.float(), dim=-1)  # (V, 768)

        n_base         = len(self.base_ids)
        n_per_class    = max(1, self.n_sample // n_base)
        sampled_idx    = []
        sampled_labels = []

        for cls_id in self.base_ids:
            mask = (vox_sem == cls_id)
            n_avail = int(mask.sum().item())
            if n_avail == 0:
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            if n_avail > n_per_class:
                perm = torch.randperm(n_avail, device=device)[:n_per_class]
                idx  = idx[perm]
            sampled_idx.append(idx)
            sampled_labels.extend([cls_id] * len(idx))

        if not sampled_idx:
            return backbone_feats.new_zeros(())

        all_idx    = torch.cat(sampled_idx, dim=0)                          # (n,)
        all_labels = torch.tensor(sampled_labels, dtype=torch.long, device=device)  # (n,)

        sampled_feats = feats[all_idx]   # (n, 768)

        # Logit matrix: each row is [proto_0..15 | neg_0..M-1] similarities
        all_keys = torch.cat([self.protos, self.neg_bank], dim=0)           # (16+M, 768)
        logits   = (sampled_feats @ all_keys.T) / self.temperature          # (n, 16+M)

        # Target = column index of the correct class prototype (first 16 columns)
        return F.cross_entropy(logits, all_labels)
