"""Model loaders with Streamlit caching.

Uses st.cache_resource (Streamlit >= 1.18) with a transparent fallback to
st.cache(allow_output_mutation=True) for older installations (Python 3.8 HPC).

All cache-keyed arguments are plain strings / primitives to ensure hashability.
"""

import os
import sys

import torch
import yaml
import streamlit as st


def _resource_cache(func):
    """Portable cache decorator for heavy objects (models, tokenizers).

    st.cache_resource  — Streamlit >= 1.18 (recommended)
    st.cache           — Streamlit <  1.18 (legacy fallback)
    """
    if hasattr(st, "cache_resource"):
        return st.cache_resource(func)
    # Legacy path — suppress the deprecation warning
    return st.cache(allow_output_mutation=True, suppress_st_warning=True)(func)


# ── Backbone ──────────────────────────────────────────────────────────────────

@_resource_cache
def load_backbone(ckpt_path, device_str):
    """Load and freeze the DisNet (MinkUNet18A) backbone.

    Args:
        ckpt_path:  absolute path to nuscenes_openseg.pth.tar
        device_str: e.g. "cuda" or "cpu"

    Returns:
        frozen DisNet model on device
    """
    import MinkowskiEngine  # noqa — verify ME is importable before building model
    from models.disnet import DisNet, state_dict_remove_moudle

    device = torch.device(device_str)

    class _Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"

    model = DisNet(_Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"[Backbone] DisNet loaded from {ckpt_path} → frozen.", flush=True)
    return model


# ── Decoder ───────────────────────────────────────────────────────────────────

@_resource_cache
def load_decoder(config_path, decoder_ckpt_path, device_str):
    """Load PanopticDecoder with CLIP token head from a training checkpoint.

    Args:
        config_path:       absolute path to demo/config.yaml (for arch params)
        decoder_ckpt_path: absolute path to best.pth
        device_str:        e.g. "cuda" or "cpu"

    Returns:
        PanopticDecoder in eval mode on device
    """
    from panoptic.models.mask_decoder import PanopticDecoder

    device = torch.device(device_str)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    decoder = PanopticDecoder(
        backbone_dim=mcfg["backbone_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_classes=mcfg["num_classes"],
        num_thing_queries=mcfg["num_thing_queries"],
        num_stuff_classes=mcfg["num_stuff_classes"],
        nheads=mcfg["nheads"],
        dim_feedforward=mcfg["dim_feedforward"],
        num_blocks=mcfg["num_blocks"],
        num_levels=mcfg["num_levels"],
        dropout=mcfg["dropout"],
        use_token_head=mcfg.get("use_token_head", True),
        token_dim=mcfg.get("token_dim", 768),
    ).to(device)

    ckpt = torch.load(decoder_ckpt_path, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder.eval()

    n_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"[Decoder] Loaded from {decoder_ckpt_path} ({n_params:.1f}M params).", flush=True)
    return decoder


# ── CLIP ──────────────────────────────────────────────────────────────────────

@_resource_cache
def load_clip_model(clip_model_name, device_str):
    """Load CLIP model for text encoding (ViT-L/14@336px).

    Args:
        clip_model_name: e.g. "ViT-L/14@336px"
        device_str:      e.g. "cuda" or "cpu"

    Returns:
        (clip_model, preprocess) tuple; preprocess not used for text-only encoding
    """
    import clip

    device = torch.device(device_str)
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model.eval()

    print(f"[CLIP] {clip_model_name} loaded on {device_str}.", flush=True)
    return model, preprocess


# ── Text prototypes ───────────────────────────────────────────────────────────

@_resource_cache
def load_text_prototypes(proto_path, device_str):
    """Load pre-computed CLIP text prototype embeddings.

    The .pt file must contain key "embeddings": (16, 768) L2-normalized float32.

    Returns:
        embeddings:  (16, 768) tensor on device
        class_names: list of 16 class name strings (or None if absent)
    """
    device = torch.device(device_str)
    data = torch.load(proto_path, map_location="cpu")
    embeddings = data["embeddings"].float().to(device)   # (16, 768)
    class_names = data.get("class_names", None)

    print(
        f"[Prototypes] {embeddings.shape} loaded from {proto_path}.",
        flush=True,
    )
    return embeddings, class_names
