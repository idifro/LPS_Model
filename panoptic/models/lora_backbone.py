"""LoRA Sparse Conv Adapters for MinkowskiEngine backbone (B1).

Inserts trainable low-rank residual adapters after frozen backbone decoder blocks.
Uses standard nn.Linear (NOT ME ops) so backward is safe — confirmed by ME gradient check.

Usage:
    from panoptic.models.lora_backbone import insert_lora_adapters

    backbone = load_backbone(...)  # frozen DisNet
    lora_adapters = insert_lora_adapters(backbone, rank=16)
    # lora_adapters is nn.ModuleDict — add to optimizer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


class SparseLoRAAdapter(nn.Module):
    """Trainable residual adapter for ME SparseTensor outputs.

    Receives SparseTensor.F (dense feature matrix), applies low-rank projection,
    returns new SparseTensor with modified features but identical coordinates.

    The backward pass goes through standard nn.Linear ops only — no ME hash map
    traversal, no atomic operations, no race conditions.

    Args:
        channels: feature dimension of the backbone block output
        rank: low-rank bottleneck dimension (default 16)
        dropout: dropout on adapter output for regularization
    """

    def __init__(self, channels: int, rank: int = 16, dropout: float = 0.05):
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        self.drop = nn.Dropout(p=dropout)

        # Kaiming init for down, zero init for up → identity at epoch 0
        nn.init.kaiming_uniform_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """Apply LoRA adapter.

        Args:
            x: ME.SparseTensor with features (V, C)

        Returns:
            ME.SparseTensor with adapted features (V, C), same coordinates
        """
        if ME is not None and isinstance(x, ME.SparseTensor):
            feats = x.F  # (V, C) dense feature matrix
            delta = self.drop(self.up(F.relu(self.down(feats))))  # (V, C)
            adapted = feats + delta  # (V, C) — identity at epoch 0 (up is zero-init)
            return ME.SparseTensor(
                features=adapted,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
        else:
            # Fallback for dense tensors (testing without ME)
            feats = x
            delta = self.drop(self.up(F.relu(self.down(feats))))
            return feats + delta


# MinkUNet18A decoder block output channels (confirmed from gradient check)
TARGET_LAYERS = {
    "net3d.block5": 128,   # decoder block (stride 8)
    "net3d.block6": 128,   # decoder block (stride 4)
    "net3d.block7": 96,    # decoder block (stride 2)
    "net3d.block8": 96,    # last decoder block (stride 1)
    "net3d.final": 768,    # output 1x1 conv
}


def insert_lora_adapters(backbone, rank: int = 16, dropout: float = 0.05,
                         target_layers: dict = None):
    """Insert LoRA adapters after specified backbone layers.

    The backbone remains fully frozen — only adapters are trainable.
    Adapters are applied via forward hooks that intercept block outputs.

    Args:
        backbone: DisNet model (frozen)
        rank: adapter bottleneck dimension
        dropout: adapter dropout rate
        target_layers: dict of {layer_name: out_channels}. Defaults to MinkUNet18A decoder.

    Returns:
        lora_adapters: nn.ModuleDict of adapters (add to optimizer)
        hooks: list of hook handles (keep alive during training)
    """
    if target_layers is None:
        target_layers = TARGET_LAYERS

    # Ensure backbone is frozen
    for p in backbone.parameters():
        p.requires_grad = False

    adapters = nn.ModuleDict()
    hooks = []

    for layer_name, out_channels in target_layers.items():
        # Create adapter
        adapter = SparseLoRAAdapter(channels=out_channels, rank=rank, dropout=dropout)
        key = layer_name.replace(".", "_")
        adapters[key] = adapter

        # Get the target module
        parts = layer_name.split(".")
        module = backbone
        for part in parts:
            module = getattr(module, part)

        # Register forward hook that applies the adapter
        def make_hook(adpt):
            def hook_fn(module, input, output):
                return adpt(output)
            return hook_fn

        handle = module.register_forward_hook(make_hook(adapter))
        hooks.append(handle)

        print(f"  LoRA adapter: {layer_name} (channels={out_channels}, rank={rank})")

    total_params = sum(p.numel() for p in adapters.parameters())
    print(f"LoRA adapters inserted: {len(adapters)} adapters, "
          f"{total_params:,} trainable params")

    return adapters, hooks


def remove_lora_hooks(hooks):
    """Remove all LoRA forward hooks (for clean inference without adapters)."""
    for h in hooks:
        h.remove()
