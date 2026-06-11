"""Script A: Novel Camera CLIP Coverage Audit.

Iterates over all precomputed .npz files and measures:
- True object coverage: valid GT instances / total GT instances (using obj_valid flag)
- Feature quality: cosine sim to own text prototype, margin_to_base = sim_own - sim_best_base
- Confusion breakdown: misclassified as base vs misclassified as another novel
- Voxel coverage: n_valid / n_voxels (real projection rate, not padded count)

Requires .npz files generated with the updated precompute script that saves:
  obj_valid (N,) bool, n_valid int, n_voxels int

Usage:
    python panoptic/scripts/diag_clip_coverage.py \
        --clip_feat_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features \
        --version v1.0-mini \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
        --out_dir panoptic/diagnostics
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

NOVEL_IDS = [2, 5, 6, 15]
BASE_IDS  = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]

WARN_VALID_THRESH    = 0.60   # flag if coverage < 60%
WARN_CONFUSED_THRESH = 0.30   # flag if confusion > 30%
MARGIN_DANGER_THRESH = 0.0    # margin_to_base < 0 → base class wins
MARGIN_WARN_THRESH   = 0.05   # margin_to_base < 0.05 → barely usable


def load_prototypes(proto_path):
    data = torch.load(proto_path, map_location="cpu")
    if isinstance(data, dict):
        embeddings  = data["embeddings"].float()
        class_names = data.get("class_names", [str(i) for i in range(embeddings.shape[0])])
    else:
        embeddings  = data.float()
        class_names = [str(i) for i in range(embeddings.shape[0])]
    protos = F.normalize(embeddings, dim=-1)
    return protos, class_names


def audit_directory(npz_dir, protos, class_names):
    protos_np = protos.numpy()  # (16, 768)

    stats = {
        cls_id: {
            "total_gt":     0,   # all GT instances (valid + invisible)
            "valid":        0,   # camera-visible instances
            "sim_own_sum":  0.0,
            "sim_own_vals": [],
            "margin_sum":   0.0,  # sim_own - sim_best_base
            "margin_vals":  [],
            "confused_base":   0,  # predicted = a base class
            "confused_novel":  0,  # predicted = a different novel class
        }
        for cls_id in range(16)
    }

    vox_frac_list = []   # n_valid / n_voxels per sample
    legacy_count  = 0    # .npz files missing new fields

    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    if not files:
        print(f"  WARNING: no .npz files found in {npz_dir}")
        return stats, vox_frac_list, legacy_count

    for fname in tqdm(files, desc=f"  Auditing {os.path.basename(npz_dir)}"):
        path = os.path.join(npz_dir, fname)
        try:
            d = np.load(path)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue

        obj_feats   = d["obj_feats"].astype(np.float32)    # (N, 768)
        obj_classes = d["obj_classes"].astype(np.int32)    # (N,)

        # True coverage requires obj_valid (new schema)
        if "obj_valid" in d:
            obj_valid = d["obj_valid"].astype(bool)        # (N,)
        else:
            legacy_count += 1
            # Old schema: all saved instances were already visible → fake 100%
            obj_valid = (np.linalg.norm(obj_feats, axis=1) > 0.1)

        # Voxel coverage
        if "n_valid" in d and "n_voxels" in d:
            n_valid  = int(d["n_valid"])
            n_voxels = int(d["n_voxels"])
            if n_voxels > 0:
                vox_frac_list.append(n_valid / n_voxels)
        # (skip if legacy)

        for i in range(len(obj_classes)):
            cls = int(obj_classes[i])
            if cls < 0 or cls >= 16:
                continue

            stats[cls]["total_gt"] += 1

            if not obj_valid[i]:
                continue

            stats[cls]["valid"] += 1

            feat = obj_feats[i]
            feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
            sim_all   = protos_np @ feat_norm    # (16,)
            predicted = int(np.argmax(sim_all))
            sim_own   = float(sim_all[cls])

            base_sims     = [sim_all[b] for b in BASE_IDS if b != cls]
            sim_best_base = float(max(base_sims)) if base_sims else 0.0
            margin        = sim_own - sim_best_base

            stats[cls]["sim_own_sum"]  += sim_own
            stats[cls]["sim_own_vals"].append(sim_own)
            stats[cls]["margin_sum"]   += margin
            stats[cls]["margin_vals"].append(margin)

            if predicted != cls:
                if predicted in BASE_IDS:
                    stats[cls]["confused_base"]  += 1
                else:
                    stats[cls]["confused_novel"] += 1

    return stats, vox_frac_list, legacy_count


def print_table(stats, class_names, group_ids, group_label, novel_ids):
    print(f"\n=== {group_label} ===")
    header = (
        f"{'Class':<25} | {'GT-inst':>8} | {'Valid%':>7} | "
        f"{'sim-own':>7} | {'margin':>7} | {'Conf-base%':>10} | {'Conf-novel%':>11}"
    )
    print(header)
    print("-" * len(header))

    warnings = []
    results  = {}

    for cls_id in group_ids:
        name     = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        s        = stats[cls_id]
        total    = s["total_gt"]
        valid    = s["valid"]
        confused_base  = s["confused_base"]
        confused_novel = s["confused_novel"]
        confused_total = confused_base + confused_novel

        valid_pct      = 100.0 * valid / total if total > 0 else 0.0
        avg_sim_own    = s["sim_own_sum"] / valid if valid > 0 else 0.0
        avg_margin     = s["margin_sum"]  / valid if valid > 0 else 0.0
        conf_base_pct  = 100.0 * confused_base  / valid if valid > 0 else 0.0
        conf_novel_pct = 100.0 * confused_novel / valid if valid > 0 else 0.0
        conf_total_pct = 100.0 * confused_total / valid if valid > 0 else 0.0

        flags = []
        if cls_id in novel_ids:
            if valid_pct < WARN_VALID_THRESH * 100:
                flags.append("LOW-COV")
                warnings.append(f"{name}: Valid%={valid_pct:.1f}% < {WARN_VALID_THRESH*100:.0f}%")
            if avg_margin < MARGIN_DANGER_THRESH:
                flags.append("MARGIN-NEG")
                warnings.append(f"{name}: margin_to_base={avg_margin:.4f} < 0 (base wins on average)")
            elif avg_margin < MARGIN_WARN_THRESH:
                flags.append("MARGIN-WARN")
                warnings.append(f"{name}: margin_to_base={avg_margin:.4f} < 0.05 (barely usable)")
            if conf_total_pct > WARN_CONFUSED_THRESH * 100:
                flags.append("HIGH-CONF")
                warnings.append(f"{name}: Confused%={conf_total_pct:.1f}% > {WARN_CONFUSED_THRESH*100:.0f}%")

        flag_str = "  ← " + " | ".join(flags) if flags else ""
        marker = "*" if cls_id in novel_ids else " "
        print(
            f"{marker}{name:<24} | {total:>8,} | {valid_pct:>6.1f}% | "
            f"{avg_sim_own:>7.4f} | {avg_margin:>+7.4f} | "
            f"{conf_base_pct:>9.1f}% | {conf_novel_pct:>10.1f}%{flag_str}"
        )

        results[cls_id] = {
            "name":           name,
            "total_gt":       total,
            "valid":          valid,
            "valid_pct":      round(valid_pct, 2),
            "avg_sim_own":    round(avg_sim_own, 4),
            "avg_margin":     round(avg_margin, 4),
            "conf_base_pct":  round(conf_base_pct, 2),
            "conf_novel_pct": round(conf_novel_pct, 2),
            "conf_total_pct": round(conf_total_pct, 2),
        }

    return results, warnings


def print_margin_histogram(margin_vals, class_name, bins=10):
    if not margin_vals:
        return
    arr  = np.array(margin_vals)
    lo   = max(-0.5, arr.min() - 0.05)
    hi   = min(1.0,  arr.max() + 0.05)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    total = len(arr)
    neg_pct = 100.0 * (arr < 0).sum() / total
    print(f"  {class_name} margin_to_base histogram  ({neg_pct:.1f}% negative = base wins):")
    for i in range(bins):
        bar   = "#" * int(30 * counts[i] / max(counts))
        label = " ← NEG" if edges[i + 1] <= 0 else ""
        print(f"    [{edges[i]:+.3f} – {edges[i+1]:+.3f}]  {bar}  ({counts[i]}){label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_feat_dir", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features")
    parser.add_argument("--version",       default="v1.0-mini")
    parser.add_argument("--proto_path",    default="panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")
    parser.add_argument("--out_dir",       default="panoptic/diagnostics")
    parser.add_argument("--histogram",     action="store_true", help="Print per-novel-class margin histograms")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npz_dir = os.path.join(args.clip_feat_dir, args.version)

    print(f"Loading prototypes from {args.proto_path}...")
    protos, class_names = load_prototypes(args.proto_path)
    print(f"  {len(class_names)} classes, embedding dim {protos.shape[1]}")
    for i, n in enumerate(class_names):
        marker = "* NOVEL" if i in NOVEL_IDS else "  base "
        print(f"    [{i:2d}] {marker}: {n}")

    print(f"\nAuditing {npz_dir}...")
    stats, vox_frac_list, legacy_count = audit_directory(npz_dir, protos, class_names)

    if legacy_count:
        print(f"\n  WARNING: {legacy_count} files missing obj_valid/n_voxels (old schema).")
        print("  Coverage% and voxel stats will be inaccurate for those files.")
        print("  Re-run precompute_clip_voxel_features.py to regenerate with new schema.")

    novel_results, novel_warnings = print_table(
        stats, class_names, NOVEL_IDS, "Novel Class Coverage", NOVEL_IDS)
    base_results, _ = print_table(
        stats, class_names, BASE_IDS, "Base Class Coverage (reference)", NOVEL_IDS)

    print("\n=== Voxel-Level Coverage (n_valid / n_voxels per sample) ===")
    if vox_frac_list:
        arr = np.array(vox_frac_list)
        print(f"  Mean: {arr.mean()*100:.1f}%   Median: {np.median(arr)*100:.1f}%   "
              f"Min: {arr.min()*100:.1f}%   Max: {arr.max()*100:.1f}%")
        low = (arr < 0.5).sum()
        print(f"  Samples with <50% voxel coverage: {low} / {len(arr)} ({100*low/len(arr):.1f}%)")
    else:
        print("  No voxel coverage data (missing n_valid/n_voxels — re-run precompute).")

    if args.histogram:
        print("\n=== Novel Class Margin-to-Base Histograms ===")
        for cls_id in NOVEL_IDS:
            if cls_id < len(class_names):
                print_margin_histogram(stats[cls_id]["margin_vals"], class_names[cls_id])

    print("\n=== VERDICT ===")
    if novel_warnings:
        print("  WARNINGS (novel classes failing gates):")
        for w in novel_warnings:
            print(f"    - {w}")
        print()
        seen = set()
        for w in novel_warnings:
            if "LOW-COV" in w or "Valid%" in w:
                k = "coverage"
                if k not in seen:
                    print("  → Low coverage: relax occlusion threshold or use depth-completion in precompute")
                    seen.add(k)
            if "MARGIN" in w:
                k = "margin"
                if k not in seen:
                    print("  → Negative/small margin: camera CLIP features misaligned with text prototype")
                    print("    Check prompt engineering; consider per-class inference logit offsets")
                    seen.add(k)
            if "HIGH-CONF" in w or "Confused%" in w:
                k = "confusion"
                if k not in seen:
                    print("  → High confusion: check whether confusion is base (prompt issue) or novel (inter-novel overlap)")
                    seen.add(k)
    else:
        print("  All novel classes pass gates (Valid%>60%, margin>0.05, Confused%<30%)")
        print("  Coverage is not the bottleneck for novel PQ=0.")

    out = {
        "version":  args.version,
        "gates": {
            "min_valid_pct":     WARN_VALID_THRESH * 100,
            "max_confused_pct":  WARN_CONFUSED_THRESH * 100,
            "min_margin":        MARGIN_WARN_THRESH,
        },
        "novel":    novel_results,
        "base":     base_results,
        "warnings": novel_warnings,
        "vox_coverage": {
            "frac_mean":   float(np.mean(vox_frac_list)) if vox_frac_list else None,
            "frac_median": float(np.median(vox_frac_list)) if vox_frac_list else None,
            "n_samples":   len(vox_frac_list),
            "legacy_files": legacy_count,
        },
    }
    out_path = os.path.join(args.out_dir, f"coverage_{args.version.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
