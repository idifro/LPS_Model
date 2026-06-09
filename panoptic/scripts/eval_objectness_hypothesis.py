"""Objectness Suppression Hypothesis Test.

Tests whether novel class embeddings exist but are suppressed by the objectness head.

For each checkpoint (Job 1a and Job 1b):
  1. Run panoptic inference WITH objectness enabled (current behavior)
  2. Run panoptic inference WITHOUT objectness (1.0 for all queries)
  3. Compare PQ_novel: if delta > 0.01, novel embeddings are being suppressed
  4. Query-level argmax distribution (are any queries ever predicting novel classes?)
  5. Zero-shot voxel counts for out-of-vocabulary concepts (trash_can, ambulance, etc.)

Usage:
    python panoptic/scripts/eval_objectness_hypothesis.py \
        --ckpt_job1a panoptic/checkpoints_job1a/best.pth \
        --cfg_job1a panoptic/configs/job1a.yaml \
        --ckpt_job1b panoptic/checkpoints_job1b/best.pth \
        --cfg_job1b panoptic/configs/job1b.yaml \
        --device cuda --split val
"""
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from panoptic.train.train_panoptic_job1 import (
    load_backbone, extract_backbone_features, get_nested
)
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.models.lora_backbone import insert_lora_adapters
from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic
from panoptic.eval.eval_pq import PanopticEval

NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

NUM_CLASSES = 16
NOVEL_IDS = [2, 5, 6, 15]   # bus, motorcycle, pedestrian, vegetation
BASE_IDS = [i for i in range(NUM_CLASSES) if i not in NOVEL_IDS]

# Out-of-vocabulary concepts not in nuScenes label set at all
ZEROSHOT_CONCEPTS = ["trash can", "fire hydrant", "ambulance", "police van"]
ZEROSHOT_PROMPTS  = [f"a {c} in a driving scene" for c in ZEROSHOT_CONCEPTS]


def query_iou_with_gt(pred_masks_vq, gt_mask_arr):
    """Per-query IoU against a single GT instance mask."""
    probs    = pred_masks_vq.sigmoid()                                         # (V, Q)
    pred_bin = probs > 0.5
    gt       = torch.as_tensor(gt_mask_arr, device=pred_masks_vq.device).bool()  # (V,)
    inter    = (pred_bin & gt[:, None]).sum(dim=0).float()
    union    = (pred_bin | gt[:, None]).sum(dim=0).float().clamp_min(1.0)
    return inter / union                                                         # (Q,)


def panoptic_inference_clip(pred_embed, pred_masks, text_protos, temperature,
                            objectness=None, min_pts=50):
    """Panoptic inference via CLIP cosine similarity, with optional objectness gating."""
    sim = pred_embed @ text_protos.T / temperature      # (Q, C)
    scores, labels = sim.max(dim=-1)                    # (Q,)

    if objectness is not None:
        scores = scores * objectness.squeeze(-1).sigmoid()

    mask_probs = pred_masks.sigmoid()                   # (V, Q)
    sorted_idx = scores.argsort(descending=True)

    V = mask_probs.shape[0]
    dev = mask_probs.device
    pan_seg = torch.zeros(V, dtype=torch.long, device=dev)
    used    = torch.zeros(V, dtype=torch.bool,  device=dev)
    instance_id = 0

    for idx in sorted_idx:
        label = labels[idx].item()
        mask  = (mask_probs[:, idx] > 0.5) & ~used
        if mask.sum() < min_pts:
            continue
        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        used[mask] = True

    return pan_seg


def evaluate_checkpoint(ckpt_path, cfg_path, device, split="mini",
                        disable_objectness=False, dataroot_mini=None,
                        zs_protos=None):
    """Full evaluation of one checkpoint.

    Returns:
        pq_mean, sq_mean, rq_mean, pq_per  — standard panoptic metrics
        pq_base, pq_novel                  — split averages
        argmax_counts (np.ndarray, 16)     — how many queries predicted each class
        zs_counts (list[int])              — voxels where each ZS concept beats all base classes
        total_voxels (int)
    """

    # ── Config ────────────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone_ckpt = cfg.get("model", {}).get("backbone_ckpt",
                                             "models/nuscenes_openseg.pth.tar")
    if not os.path.exists(backbone_ckpt):
        backbone_ckpt = f"/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/{backbone_ckpt}"
    backbone = load_backbone(backbone_ckpt, device)

    # ── Decoder ───────────────────────────────────────────────────────────────
    # enable_single_clip_head() MUST be called before load_state_dict so the
    # clip_embed_head and objectness_head modules exist in the architecture.
    decoder = PanopticDecoder(
        backbone_dim      = get_nested(cfg, "model.backbone_dim", 768),
        hidden_dim        = get_nested(cfg, "model.hidden_dim", 256),
        num_classes       = get_nested(cfg, "model.num_classes", 16),
        num_thing_queries = get_nested(cfg, "model.num_thing_queries", 100),
        num_stuff_classes = get_nested(cfg, "model.num_stuff_classes", 6),
        nheads            = get_nested(cfg, "model.nheads", 8),
        dim_feedforward   = get_nested(cfg, "model.dim_feedforward", 1024),
        num_blocks        = get_nested(cfg, "model.num_blocks", 3),
        num_levels        = get_nested(cfg, "model.num_levels", 3),
        dropout           = get_nested(cfg, "model.dropout", 0.0),
        use_token_head    = True,
        token_dim         = get_nested(cfg, "model.clip_embed_dim", 768),
    )
    if get_nested(cfg, "model.head_type", "dual") == "single_clip":
        decoder.enable_single_clip_head(
            clip_dim=get_nested(cfg, "model.clip_embed_dim", 768)
        )
    decoder = decoder.to(device)
    decoder.eval()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    decoder.load_state_dict(ckpt.get("decoder", ckpt), strict=False)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Checkpoint saves under "lora_adapters"; fall back to "lora_state" for
    # warm-start checkpoints that used an older key.
    lora_adapters = None
    if get_nested(cfg, "lora.enabled", False):
        lora_adapters, _hooks = insert_lora_adapters(
            backbone,
            rank    = get_nested(cfg, "lora.rank",    16),
            dropout = get_nested(cfg, "lora.dropout", 0.05),
        )
        lora_state = ckpt.get("lora_adapters", ckpt.get("lora_state", None))
        if lora_state is not None:
            lora_adapters.load_state_dict(lora_state, strict=False)
        else:
            print("  WARNING: No LoRA state in checkpoint — using zero-init adapters!")
        lora_adapters = lora_adapters.to(device)
        lora_adapters.eval()

    # ── Text prototypes ───────────────────────────────────────────────────────
    # Handles both raw-tensor and dict-with-"embeddings" format; always normalize.
    text_protos = None
    proto_path  = get_nested(cfg, "token.prototype_path", None)
    if proto_path:
        if not os.path.exists(proto_path):
            proto_path = f"/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/{proto_path}"
        if os.path.exists(proto_path):
            raw = torch.load(proto_path, map_location=device)
            if isinstance(raw, dict) and "embeddings" in raw:
                text_protos = raw["embeddings"].float()
            else:
                text_protos = raw.float()
            text_protos = F.normalize(text_protos, dim=-1).to(device)   # (16, 768)
        else:
            print(f"  WARNING: Prototype file not found: {proto_path}")

    # ── Temperature ───────────────────────────────────────────────────────────
    # token_criterion.state_dict() stores log_temperature (nn.Parameter).
    # Actual temperature = exp(log_temperature).clamp(1.0, 5.0) — same as
    # the property defined in the C1TokenCriterion class.
    temperature = 1.0
    if "token_criterion" in ckpt:
        tc_state = ckpt["token_criterion"]
        if isinstance(tc_state, dict) and "log_temperature" in tc_state:
            log_t = tc_state["log_temperature"]
            temperature = torch.exp(log_t).clamp(1.0, 5.0).item()
            print(f"  Temperature from checkpoint: {temperature:.4f}  (log_t={log_t.item():.4f})")
        else:
            print("  WARNING: log_temperature not found in token_criterion — using τ=1.0")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg  = cfg.get("data", {})
    if split == "mini":
        version  = "v1.0-mini"
        dataroot = dataroot_mini or data_cfg.get("dataroot")
    else:
        version  = data_cfg.get("version", "v1.0-trainval")
        dataroot = data_cfg.get("dataroot")

    dataset = NuScenesPanoptic(
        dataroot   = dataroot,
        version    = version,
        split      = "val",
        augment    = False,
        voxel_size = data_cfg.get("voxel_size", 0.05),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    evaluator     = PanopticEval(NUM_CLASSES, min_points=50)
    argmax_counts = np.zeros(NUM_CLASSES, dtype=np.int64)   # per-class query count
    zs_counts     = [0] * len(ZEROSHOT_CONCEPTS)
    total_voxels  = 0
    # Per-novel-class aggregates (only populated on WITH-objectness pass)
    novel_diag = {c: dict(n=0, found=0, top1_novel=0, top5_novel=0,
                          iou_sum=0.0, obj_sum=0.0)
                  for c in NOVEL_IDS}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"  [{batch_idx}/{len(loader)}]", end="\r")

            # Squeeze batch dim added by DataLoader
            unbatched = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            feats, coords = extract_backbone_features(
                backbone, unbatched, device, lora_adapters
            )
            outputs = decoder(feats, coords)

            gt_classes_cur = unbatched["gt_classes"].numpy()
            gt_masks_cur   = unbatched["gt_masks"].numpy()
            V              = len(unbatched["gt_sem"])

            if text_protos is None or "pred_embed" not in outputs:
                # Hard fallback — should not happen after bug fixes
                V = outputs["pred_masks"][0].shape[0]
                pan_pred = torch.zeros(V, dtype=torch.long, device=device)
            else:
                pred_embed = outputs["pred_embed"][0]   # (Q, 768)
                pred_masks = outputs["pred_masks"][0]   # (V, Q)
                obj_scores = outputs.get("pred_obj", [None])[0]  # (Q, 1) or None

                # ── Query argmax class distribution ───────────────────────────
                sim_all = pred_embed @ text_protos.T    # (Q, 16)
                argmax  = sim_all.argmax(dim=-1)        # (Q,)
                for c in range(NUM_CLASSES):
                    argmax_counts[c] += (argmax == c).sum().item()

                # ── Panoptic prediction ────────────────────────────────────────
                pan_pred = panoptic_inference_clip(
                    pred_embed, pred_masks, text_protos, temperature,
                    objectness=None if disable_objectness else obj_scores,
                )

                # ── Zero-shot voxel counts ─────────────────────────────────────
                if zs_protos is not None:
                    mask_probs    = pred_masks.sigmoid()            # (V, Q)
                    voxel_embeds  = mask_probs @ pred_embed         # (V, 768)
                    voxel_embeds  = F.normalize(voxel_embeds, dim=-1)
                    zs_sim        = voxel_embeds @ zs_protos.T      # (V, 4)
                    base_sim_max  = (voxel_embeds @ text_protos.T).max(dim=-1).values  # (V,)
                    for zi in range(len(ZEROSHOT_CONCEPTS)):
                        above = (zs_sim[:, zi] > base_sim_max).sum().item()
                        zs_counts[zi] += above
                    total_voxels += voxel_embeds.shape[0]

                # ── Novel GT query diagnostic (WITH objectness pass only) ──────
                if not disable_objectness:
                    for m_idx, cls in enumerate(gt_classes_cur):
                        cls = int(cls)
                        if cls not in NOVEL_IDS:
                            continue
                        ious    = query_iou_with_gt(pred_masks, gt_masks_cur[m_idx])
                        best_q  = int(ious.argmax().item())
                        best_iou = float(ious[best_q].item())

                        sim = pred_embed[best_q] @ text_protos.T / temperature
                        topv, topi = sim.topk(5)
                        top5_ids = [int(i) for i in topi]

                        obj = None
                        if obj_scores is not None:
                            obj = float(obj_scores[best_q].squeeze(-1).sigmoid().item())

                        d = novel_diag[cls]
                        d["n"]       += 1
                        d["iou_sum"] += best_iou
                        if best_iou > 0.1:
                            d["found"] += 1
                        if top5_ids[0] == cls:
                            d["top1_novel"] += 1
                        if cls in top5_ids:
                            d["top5_novel"] += 1
                        if obj is not None:
                            d["obj_sum"] += obj

                        obj_str = f"{obj:.3f}" if obj is not None else "n/a"
                        # print(f"  [novel-diag s{batch_idx} m{m_idx}] "
                        #       f"gt={NUSCENES_CLASS_NAMES[cls]}  "
                        #       f"iou={best_iou:.3f}  "
                        #       f"obj={obj_str}  "
                        #       f"top5={[(NUSCENES_CLASS_NAMES[int(i)], round(float(v),3)) for v,i in zip(topv,topi)]}")

            # ── GT panoptic encoding ───────────────────────────────────────────
            gt_pan = np.zeros(V, dtype=np.int64)
            for m_idx, cls in enumerate(gt_classes_cur):
                gt_pan[gt_masks_cur[m_idx]] = int(cls) * 1000 + (m_idx + 1)

            pred_np = pan_pred.cpu().numpy()
            valid   = unbatched["gt_sem"].numpy() != 255
            evaluator.addBatch(
                (pred_np[valid] // 1000).astype(np.int32),
                (pred_np[valid] %  1000).astype(np.int32),
                (gt_pan[valid]  // 1000).astype(np.int32),
                (gt_pan[valid]  %  1000).astype(np.int32),
            )

    print()

    pq_mean, sq_mean, rq_mean, pq_per, _, _ = evaluator.getPQ()
    pq_base  = np.mean([pq_per[i] for i in BASE_IDS  if i < len(pq_per)])
    pq_novel = np.mean([pq_per[i] for i in NOVEL_IDS if i < len(pq_per)])

    # Print novel-GT query diagnostic summary (only present on WITH-obj pass)
    if any(v["n"] > 0 for v in novel_diag.values()):
        print(f"\n  Novel GT query diagnostic (WITH objectness, {len(loader)} samples):")
        print(f"    {'Class':<22} {'GT-inst':>8} {'IoU>0.1':>8} {'top1-nov':>9} {'top5-nov':>9} {'avg-IoU':>8} {'avg-obj':>8}")
        print(f"    {'-'*72}")
        for c in NOVEL_IDS:
            d = novel_diag[c]
            n = max(d["n"], 1)
            print(f"    {NUSCENES_CLASS_NAMES[c]:<22} {d['n']:>8} {d['found']:>8} "
                  f"{d['top1_novel']:>9} {d['top5_novel']:>9} "
                  f"{d['iou_sum']/n:>8.3f} {d['obj_sum']/n:>8.3f}")

    return (pq_mean, sq_mean, rq_mean, pq_per,
            pq_base, pq_novel,
            argmax_counts, zs_counts, total_voxels, novel_diag)


def load_zs_protos(device):
    """Encode zero-shot concept prompts with CLIP ViT-L/14."""
    try:
        import clip as _clip
        print(f"Loading CLIP for zero-shot concepts: {ZEROSHOT_CONCEPTS}")
        model, _ = _clip.load("ViT-L/14", device=device, jit=False)
        model.eval()
        tokens = _clip.tokenize(ZEROSHOT_PROMPTS).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens).float()
            emb = F.normalize(emb, dim=-1)
        del model
        torch.cuda.empty_cache()
        return emb   # (4, 768)
    except Exception as e:
        print(f"  WARNING: Could not load CLIP for zero-shot protos: {e}")
        return None


def print_results(label, pq_all, pq_base, pq_novel, pq_per,
                  argmax_counts, zs_counts, total_voxels, with_obj):
    tag = "WITH objectness" if with_obj else "WITHOUT objectness"
    print(f"\n  [{tag}]")
    print(f"    PQ_all={pq_all:.4f}  PQ_base={pq_base:.4f}  PQ_novel={pq_novel:.4f}")

    print(f"\n  Query argmax distribution ({tag}):")
    print(f"    {'Class':<22} {'Queries':>8}  {'Novel?':>6}")
    print(f"    {'-'*42}")
    for c, name in enumerate(NUSCENES_CLASS_NAMES):
        novel_tag = " <-- NOVEL" if c in NOVEL_IDS else ""
        print(f"    {name:<22} {argmax_counts[c]:>8}{novel_tag}")
    total_queries = argmax_counts.sum()
    novel_queries = sum(argmax_counts[c] for c in NOVEL_IDS)
    print(f"    {'TOTAL':<22} {total_queries:>8}")
    print(f"    Novel-class queries: {novel_queries} / {total_queries} "
          f"({100*novel_queries/max(total_queries,1):.1f}%)")

    if total_voxels > 0:
        print(f"\n  Zero-shot voxel counts ({tag})  [voxels where ZS sim > max base sim]:")
        print(f"    {'Concept':<20} {'Voxels':>10}  {'%':>6}")
        print(f"    {'-'*42}")
        for zi, concept in enumerate(ZEROSHOT_CONCEPTS):
            pct = 100.0 * zs_counts[zi] / total_voxels
            print(f"    {concept:<20} {zs_counts[zi]:>10}  {pct:>5.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_job1a",    required=True)
    parser.add_argument("--cfg_job1a",     required=True)
    parser.add_argument("--ckpt_job1b",    required=True)
    parser.add_argument("--cfg_job1b",     required=True)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--split",         default="mini", choices=["mini", "val"])
    parser.add_argument("--dataroot_mini", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("OBJECTNESS SUPPRESSION HYPOTHESIS TEST")
    print("=" * 80)

    # Encode zero-shot concept protos once (shared across all evaluations)
    zs_protos = load_zs_protos(device)

    results = {}

    for job_name, ckpt_path, cfg_path in [
        ("job1a", args.ckpt_job1a, args.cfg_job1a),
        ("job1b", args.ckpt_job1b, args.cfg_job1b),
    ]:
        label_map = {
            "job1a": "JOB 1a: Token Head Alignment (No Distillation)",
            "job1b": "JOB 1b: Camera CLIP Distillation (L_O + L_V)",
        }
        print(f"\n{'=' * 80}")
        print(label_map[job_name])
        print("=" * 80)

        all_res = {}
        for disable_obj in [False, True]:
            tag = "WITH" if not disable_obj else "WITHOUT"
            print(f"\n[{job_name.upper()}] {tag} objectness...")
            (pq_all, sq, rq, pq_per,
             pq_base, pq_novel,
             argmax_counts, zs_counts, total_voxels, novel_diag) = evaluate_checkpoint(
                ckpt_path, cfg_path, device,
                split             = args.split,
                disable_objectness= disable_obj,
                dataroot_mini     = args.dataroot_mini,
                zs_protos         = zs_protos,
            )
            all_res[tag] = dict(
                pq_all=pq_all, pq_base=pq_base, pq_novel=pq_novel, pq_per=pq_per,
                argmax_counts=argmax_counts, zs_counts=zs_counts,
                total_voxels=total_voxels, novel_diag=novel_diag,
            )
            print_results(job_name, pq_all, pq_base, pq_novel, pq_per,
                          argmax_counts, zs_counts, total_voxels,
                          with_obj=(not disable_obj))

        delta = all_res["WITHOUT"]["pq_novel"] - all_res["WITH"]["pq_novel"]
        results[job_name] = {**all_res, "delta": delta}

        print(f"\n  ── Summary: {job_name.upper()} ──")
        print(f"  WITH objectness:    PQ_novel = {all_res['WITH']['pq_novel']:.4f}")
        print(f"  WITHOUT objectness: PQ_novel = {all_res['WITHOUT']['pq_novel']:.4f}")
        print(f"  Delta (PQ_novel):   {delta:+.4f}")

        novel_q_with    = sum(all_res['WITH']['argmax_counts'][c]    for c in NOVEL_IDS)
        novel_q_without = sum(all_res['WITHOUT']['argmax_counts'][c] for c in NOVEL_IDS)
        total_q         = all_res['WITH']['argmax_counts'].sum()
        print(f"  Novel-class queries (WITH obj):    {novel_q_with} / {total_q}")
        print(f"  Novel-class queries (WITHOUT obj): {novel_q_without} / {total_q}")

        if delta > 0.01:
            print(f"  FINDING: Novel embeddings exist but ARE suppressed by objectness")
        elif novel_q_with == 0 and novel_q_without == 0:
            print(f"  FINDING: Zero novel-class queries — token head never predicts novel classes")
            print(f"           Likely cause: InfoNCE denominator pushed embeddings away from novel protos")
        elif delta < 0.005:
            print(f"  FINDING: Objectness is NOT the bottleneck for novel predictions")
        else:
            print(f"  FINDING: Partial — weak novel embeddings present")

    # ── Final diagnosis ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL DIAGNOSIS")
    print("=" * 80)

    job1b = results["job1b"]
    delta = job1b["delta"]
    novel_q = sum(job1b["WITHOUT"]["argmax_counts"][c] for c in NOVEL_IDS)

    if delta > 0.01:
        print("CONFIRMED: Novel embeddings suppressed by objectness → proceed with Option 4 fix")
    elif novel_q == 0:
        print("NOT CONFIRMED (no novel queries): Job 1b did not produce detectable novel")
        print("panoptic predictions under this inference procedure.")
        print()
        print("This does NOT mean novel embeddings are absent at the feature level.")
        print("Run query-level embedding diagnostics (eval_voxel_margin.py on Job 1b)")
        print("before concluding distillation failed.")
        print()
        print("Most likely causes (in priority order):")
        print("  1. InfoNCE denominator: novel protos as negatives pushed embeddings away")
        print("     → Fix: remove novel from InfoNCE denominator in Job 1a v2")
        print("  2. L_V weight too low (0.1 vs token weight 1.0) — token loss dominates")
        print("     → Fix: increase L_V weight to 0.5–1.0 in Job 1b v2")
        print("  3. L_O only supervises base-matched queries; novel queries get no direct pull")
        print("     → Fix: include novel instances in L_O using all GT instances, not just matched")
    elif delta < 0.005:
        print("NOT CONFIRMED: Objectness is not the bottleneck.")
        print("Novel predictions are absent regardless of objectness gate.")
        print("Investigate distillation quality and embedding space alignment.")
    else:
        print("PARTIAL: Weak novel signal. Investigate before committing to fix.")

    print()
    print("Zero-shot OOV voxels (Job 1b, WITHOUT objectness):")
    r = job1b["WITHOUT"]
    if r["total_voxels"] > 0:
        for zi, concept in enumerate(ZEROSHOT_CONCEPTS):
            pct = 100.0 * r["zs_counts"][zi] / r["total_voxels"]
            print(f"  {concept:<20}: {r['zs_counts'][zi]:>8} voxels ({pct:.3f}%)")
    else:
        print("  (zero-shot protos not available)")

    print("=" * 80)


if __name__ == "__main__":
    exit(main())
