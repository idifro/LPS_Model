"""NuScenes Panoptic CLIP Demo — Streamlit application.

Usage (from repo root on HPC):
    streamlit run app.py [-- --config demo/config.yaml]

Port forwarding for remote HPC:
    ssh -L 8501:localhost:8501 <hpc-user>@<hpc-host>
    Then open http://localhost:8501 in your browser.

Layout:
    Sidebar  — data navigation, text prompt, threshold slider, options
    Tab 1    — Preview: raw 6-camera images (updates on sample change)
    Tab 2    — Inference Results: text-query overlay + all-classes overlay + BEV
    Tab 3    — Query Scores: bar chart of cosine similarities for active queries
"""

import os
import sys

# ── Ensure repo root is on sys.path (so 'models/' and 'panoptic/' are importable) ──
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NuScenes Panoptic CLIP Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(_REPO_ROOT, "demo", "config.yaml")
with open(_CFG_PATH, "r") as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(path):
    """Resolve a path relative to repo root if not already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


BACKBONE_CKPT  = _resolve(_CFG["model"]["backbone_ckpt"])
DECODER_CKPT   = _resolve(_CFG["checkpoint"]["decoder_ckpt"])
PROTO_PATH     = _resolve(_CFG["token"]["prototype_path"])
DATA_ROOT      = _CFG["data"]["dataroot"]
NUSC_VERSION   = _CFG["data"]["version"]
VOXEL_SIZE     = float(_CFG["data"]["voxel_size"])
POINT_RANGE    = np.array(_CFG["data"]["point_range"], dtype=np.float32)
CLIP_MODEL     = _CFG["token"]["clip_model"]
OVERLAP_THR    = float(_CFG["inference"]["overlap_thr"])
ENSEMBLE_ALPHA = float(_CFG["inference"]["ensemble_alpha"])
MIN_MASK_AREA  = int(_CFG["demo"]["min_mask_area"])
DEFAULT_THR    = float(_CFG["demo"]["default_threshold"])
POINT_SIZE     = int(_CFG["demo"]["point_size"])
ALPHA          = float(_CFG["demo"]["alpha"])
MAX_DEPTH      = float(_CFG["demo"]["max_depth"])

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Demo module imports
# ─────────────────────────────────────────────────────────────────────────────
from demo.data_utils import (
    load_nusc,
    get_scenes_for_split,
    get_samples_for_scene,
    load_sample_for_inference,
)
from demo.models import (
    load_backbone,
    load_decoder,
    load_clip_model,
    load_text_prototypes,
)
from demo.inference import run_full_inference
from demo.text_query import (
    encode_text_prompt,
    find_matching_queries,
    get_all_active_query_scores,
)
from demo.visualization import (
    CAMERAS_ORDERED,
    CAM_LABELS,
    NUSCENES_CLASS_NAMES,
    RANK_COLORS,
    build_text_query_colors,
    build_class_colors,
    get_raw_camera_images,
    make_camera_overlays,
    make_bev_figure,
    make_text_query_legend_fig,
    make_class_legend_fig,
    make_query_score_fig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [
    ("inference_done",     False),
    ("inference_results",  None),
    ("current_token",      None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — data navigation and controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Panoptic CLIP Demo")
    st.caption(f"Device: **{DEVICE_STR.upper()}**  ·  Dataset: **{NUSC_VERSION}**")
    st.divider()

    # ── Data ──────────────────────────────────────────────────────────────
    st.subheader("Data")
    split = st.radio("Split", ["train", "val"], index=1, horizontal=True)

    nusc   = load_nusc(DATA_ROOT, NUSC_VERSION)
    scenes = get_scenes_for_split(nusc, NUSC_VERSION, split)

    if not scenes:
        st.error(f"No scenes found for split='{split}' in {NUSC_VERSION}.")
        st.stop()

    scene_labels = [
        "{} ({} samples)".format(name, n) for _, name, n in scenes
    ]
    scene_idx = st.selectbox(
        "Scene", range(len(scene_labels)),
        format_func=lambda i: scene_labels[i],
    )
    selected_scene_token = scenes[scene_idx][0]

    samples       = get_samples_for_scene(nusc, selected_scene_token)
    sample_labels = [
        "Sample {} / {}".format(i + 1, len(samples)) for i in range(len(samples))
    ]
    sample_idx = st.selectbox(
        "Sample", range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
    )
    selected_token = samples[sample_idx][0]

    # Reset stale inference results when the user navigates to a new sample
    if selected_token != st.session_state.current_token:
        st.session_state.current_token    = selected_token
        st.session_state.inference_done   = False
        st.session_state.inference_results = None

    st.divider()

    # ── Text Query ────────────────────────────────────────────────────────
    st.subheader("Text Query")
    text_prompt = st.text_input(
        "Prompt",
        value="a car in the scene",
        placeholder="e.g. a pedestrian in the scene",
        help="Type anything. All matching object instances are shown — "
             "no count cap. Adjust the threshold to control sensitivity.",
    )
    sim_threshold = st.slider(
        "Similarity threshold",
        min_value=0.05, max_value=1.0,
        value=DEFAULT_THR, step=0.01,
        help="Only queries with cosine similarity ≥ this value are shown. "
             "Lower → more matches; higher → fewer, more confident matches.",
    )
    run_btn = st.button("▶  Run Inference", type="primary", use_container_width=True)

    st.divider()

    # ── Options ───────────────────────────────────────────────────────────
    st.subheader("Options")
    show_all_classes = st.checkbox(
        "Show all 16 classes overlay",
        value=False,
        help="Overlay panoptic segmentation for all 16 nuScenes classes using "
             "the ensemble classifier (linear head + CLIP cosine).",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached — runs once per session, re-used across rerenders)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models (first run may take ~30–60 s on HPC) …"):
    backbone                    = load_backbone(BACKBONE_CKPT, DEVICE_STR)
    decoder                     = load_decoder(_CFG_PATH, DECODER_CKPT, DEVICE_STR)
    clip_model, _               = load_clip_model(CLIP_MODEL, DEVICE_STR)
    text_protos, proto_names    = load_text_prototypes(PROTO_PATH, DEVICE_STR)

device          = torch.device(DEVICE_STR)
text_protos_cpu = text_protos.cpu()

# ─────────────────────────────────────────────────────────────────────────────
# Inference  (triggered by the "Run Inference" button, result cached in session)
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running backbone + decoder (5–20 s) …"):
        batch = load_sample_for_inference(
            nusc, selected_token, VOXEL_SIZE, POINT_RANGE,
        )
        sem_pred, inst_pred, pred_masks, pred_tokens, pred_logits = run_full_inference(
            backbone, decoder, batch, text_protos, device,
            alpha=ENSEMBLE_ALPHA, overlap_thr=OVERLAP_THR,
        )
        st.session_state.inference_results = {
            "sem_pred":      sem_pred,                     # (V,) int64 ndarray
            "inst_pred":     inst_pred,                    # (V,) int64 ndarray
            "pred_masks":    pred_masks,                   # (V, Q) float32 CPU tensor
            "pred_tokens":   pred_tokens,                  # (Q, 768) float32 CPU tensor
            "pred_logits":   pred_logits,                  # (Q, 17) float32 CPU tensor
            "vox_feats":     batch["vox_feats"].numpy(),   # (V, 3) float32 ndarray
            "cam_paths":     batch["cam_paths"],
            "sample_record": batch["sample_record"],
        }
        st.session_state.inference_done = True
    st.toast("Inference complete!", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Main content — three tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_preview, tab_results, tab_scores = st.tabs(
    ["📷  Preview", "🎯  Inference Results", "📊  Query Scores"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Preview: raw camera images, no inference required
# ══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    st.header("Camera Preview")
    st.caption("Token: `{}`".format(selected_token))

    preview_batch = load_sample_for_inference(
        nusc, selected_token, VOXEL_SIZE, POINT_RANGE,
    )
    raw_imgs = get_raw_camera_images(preview_batch["cam_paths"])

    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    for col, cam in zip(row1_cols, CAMERAS_ORDERED[:3]):
        col.image(raw_imgs[cam], caption=CAM_LABELS[cam], use_column_width=True)
    for col, cam in zip(row2_cols, CAMERAS_ORDERED[3:]):
        col.image(raw_imgs[cam], caption=CAM_LABELS[cam], use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Inference Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.inference_done:
        st.info("Click **▶ Run Inference** in the sidebar to see segmentation results.")
    else:
        res          = st.session_state.inference_results
        raw_xyz      = res["vox_feats"]       # (V, 3) float32 ndarray
        sem_pred     = res["sem_pred"]         # (V,) int64 ndarray
        inst_pred    = res["inst_pred"]        # (V,) int64 ndarray
        pred_masks   = res["pred_masks"]       # (V, Q) CPU tensor
        pred_tokens  = res["pred_tokens"]      # (Q, 768) CPU tensor
        sample_rec   = res["sample_record"]
        V            = raw_xyz.shape[0]

        # ── Text query matching (fast — no model forward) ─────────────────
        matched_queries = []
        if text_prompt.strip():
            text_feat = encode_text_prompt(clip_model, text_prompt.strip(), device)
            matched_queries = find_matching_queries(
                text_feat, pred_tokens, pred_masks,
                threshold=sim_threshold,
                min_mask_area=MIN_MASK_AREA,
            )

        # ── Section A: Text Query Overlay ─────────────────────────────────
        st.header("Text Query Overlay")

        if text_prompt.strip():
            if matched_queries:
                st.success(
                    "Found **{} matching instance(s)** "
                    "for '{}' (threshold = {:.2f})".format(
                        len(matched_queries), text_prompt, sim_threshold,
                    )
                )
            else:
                st.warning(
                    "No instances matched '{}' above threshold {:.2f}. "
                    "Try lowering the threshold or rephrasing "
                    "(e.g. 'a car in the scene').".format(text_prompt, sim_threshold)
                )

            tq_colors   = build_text_query_colors(matched_queries, V)
            tq_overlays = make_camera_overlays(
                raw_xyz, tq_colors, nusc, sample_rec,
                max_depth=MAX_DEPTH, point_size=POINT_SIZE, alpha=ALPHA,
            )

            r1 = st.columns(3)
            r2 = st.columns(3)
            for col, cam in zip(r1, CAMERAS_ORDERED[:3]):
                col.image(tq_overlays[cam], caption=CAM_LABELS[cam], use_column_width=True)
            for col, cam in zip(r2, CAMERAS_ORDERED[3:]):
                col.image(tq_overlays[cam], caption=CAM_LABELS[cam], use_column_width=True)

            # BEV — text query
            st.subheader("BEV — Text Query")
            bev_tq = make_bev_figure(
                raw_xyz, tq_colors,
                title="BEV  ·  '{}' (thr = {:.2f})".format(text_prompt, sim_threshold),
            )
            st.pyplot(bev_tq, use_container_width=True)
            plt.close(bev_tq)

            # Legend table — one row per matched query
            st.subheader("Matched Query Legend")
            if matched_queries:
                legend_rows = []
                for rank, (qidx, score, binary_mask) in enumerate(matched_queries):
                    sim_to_protos = pred_tokens[qidx] @ text_protos_cpu.T  # (16,)
                    nearest_cls   = sim_to_protos.argmax().item()
                    nearest_sim   = sim_to_protos[nearest_cls].item()
                    color_hex     = "#{:02X}{:02X}{:02X}".format(
                        *RANK_COLORS[min(rank, len(RANK_COLORS) - 1)].tolist()
                    )
                    legend_rows.append({
                        "Rank":          rank + 1,
                        "Color":         color_hex,
                        "Score":         round(score, 4),
                        "Nearest class": NUSCENES_CLASS_NAMES[nearest_cls],
                        "Class sim":     round(nearest_sim, 3),
                        "N voxels":      int(binary_mask.sum()),
                    })

                try:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(legend_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                except ImportError:
                    # Fallback: render as markdown table
                    header = "| " + " | ".join(legend_rows[0].keys()) + " |"
                    sep    = "| " + " | ".join(["---"] * len(legend_rows[0])) + " |"
                    rows   = [
                        "| " + " | ".join(str(v) for v in row.values()) + " |"
                        for row in legend_rows
                    ]
                    st.markdown("\n".join([header, sep] + rows))
        else:
            st.info("Enter a text prompt in the sidebar to highlight matching instances.")

        # ── Section B: All 16 Classes Overlay ─────────────────────────────
        if show_all_classes:
            st.divider()
            st.header("All 16 Classes Overlay")
            st.caption(
                "Ensemble classifier (linear head + CLIP cosine). "
                "Colours match the legend below."
            )

            cls_colors   = build_class_colors(sem_pred)
            cls_overlays = make_camera_overlays(
                raw_xyz, cls_colors, nusc, sample_rec,
                max_depth=MAX_DEPTH, point_size=POINT_SIZE, alpha=ALPHA,
            )

            r1c = st.columns(3)
            r2c = st.columns(3)
            for col, cam in zip(r1c, CAMERAS_ORDERED[:3]):
                col.image(cls_overlays[cam], caption=CAM_LABELS[cam], use_column_width=True)
            for col, cam in zip(r2c, CAMERAS_ORDERED[3:]):
                col.image(cls_overlays[cam], caption=CAM_LABELS[cam], use_column_width=True)

            # BEV — all classes
            st.subheader("BEV — All Classes")
            bev_cls = make_bev_figure(raw_xyz, cls_colors, title="BEV  ·  All 16 Classes")
            st.pyplot(bev_cls, use_container_width=True)
            plt.close(bev_cls)

            # Class legend
            st.subheader("Class Legend")
            cls_legend_fig = make_class_legend_fig(sem_pred)
            st.pyplot(cls_legend_fig, use_container_width=True)
            plt.close(cls_legend_fig)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Query Scores: cosine similarity bar chart for all active queries
# ══════════════════════════════════════════════════════════════════════════════
with tab_scores:
    if not st.session_state.inference_done:
        st.info("Run inference first to see query scores.")
    elif not text_prompt.strip():
        st.info("Enter a text prompt in the sidebar to compute query scores.")
    else:
        st.header("Query Cosine Similarity Scores")
        st.caption(
            "Each bar is one **active** decoder query (mask area ≥ {} voxels). "
            "Coloured bars matched the text prompt above the threshold. "
            "The label shows the query's nearest class in CLIP-prototype space.".format(
                MIN_MASK_AREA
            )
        )

        res_t        = st.session_state.inference_results
        pred_tokens_t = res_t["pred_tokens"]   # (Q, 768) CPU tensor
        pred_masks_t  = res_t["pred_masks"]    # (V, Q) CPU tensor

        text_feat_t = encode_text_prompt(clip_model, text_prompt.strip(), device)

        active_qs, sims = get_all_active_query_scores(
            text_feat_t, pred_tokens_t, pred_masks_t,
            min_mask_area=MIN_MASK_AREA,
        )
        matched_t = find_matching_queries(
            text_feat_t, pred_tokens_t, pred_masks_t,
            threshold=sim_threshold, min_mask_area=MIN_MASK_AREA,
        )

        score_fig = make_query_score_fig(
            active_qs, sims, matched_t, sim_threshold,
            pred_tokens_cpu=pred_tokens_t,
            text_protos_cpu=text_protos_cpu,
        )
        st.pyplot(score_fig, use_container_width=True)
        plt.close(score_fig)

        n_active  = len(active_qs)
        n_matched = len(matched_t)
        st.caption(
            "**{}** active queries  ·  "
            "**{}** match '{}' (threshold = {:.2f})".format(
                n_active, n_matched, text_prompt, sim_threshold,
            )
        )
