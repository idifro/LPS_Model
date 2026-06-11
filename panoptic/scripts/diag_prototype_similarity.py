"""Script B: Text Prototype Similarity Analysis.

Measures:
- Full 16×16 cosine similarity matrix between class text prototypes
- For each novel class: proto_margin = 1 - sim(novel_proto, nearest_base_proto)
  Small margin = fragile classification even with good embedding alignment
- For each novel class: nearest negative bank entry
  High sim = the InfoNCE denominator during C1 training may interfere with
  embedding geometry for that class (does NOT affect inference argmax directly)

Usage:
    python panoptic/scripts/diag_prototype_similarity.py \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
        --neg_bank_path panoptic/clip_prototypes/negative_bank_200.pt \
        --out_dir panoptic/diagnostics
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

NOVEL_IDS = [2, 5, 6, 15]
BASE_IDS  = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]

# proto_margin = 1 - nearest_base_sim  (lower = more fragile at inference)
FRAGILE_MARGIN_THRESH = 0.15   # margin < 0.15: classification is fragile
MODERATE_MARGIN_THRESH = 0.25  # margin < 0.25: worth monitoring

# novel-neg-bank sim thresholds (training geometry only, not inference)
NEG_INTERFERE_THRESH = 0.85    # likely to distort embedding during C1 training
NEG_SOFT_THRESH      = 0.75    # mild interference risk


def load_prototypes(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        emb   = data["embeddings"].float()
        names = data.get("class_names", [str(i) for i in range(emb.shape[0])])
    else:
        emb   = data.float()
        names = [str(i) for i in range(emb.shape[0])]
    return F.normalize(emb, dim=-1), names


def load_neg_bank(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        emb      = data["embeddings"].float()
        concepts = data.get("concepts", [f"neg_{i}" for i in range(emb.shape[0])])
    else:
        emb      = data.float()
        concepts = [f"neg_{i}" for i in range(emb.shape[0])]
    return F.normalize(emb, dim=-1), concepts


def print_similarity_matrix(sim_pp, class_names):
    n = sim_pp.shape[0]
    col_w = 7
    short = [n[:6] for n in class_names]
    print("\n=== 16×16 Prototype Cosine Similarity Matrix (novel rows/cols marked *) ===")
    header = " " * 26 + "  ".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print(" " * 26 + "-" * (col_w * n + 2 * (n - 1)))
    for i in range(n):
        marker = "*" if i in NOVEL_IDS else " "
        row_label = f"{marker}{class_names[i]:<24}"
        row_vals  = "  ".join(f"{sim_pp[i, j].item():>{col_w}.3f}" for j in range(n))
        print(f"{row_label}  {row_vals}")


def analyse_novel_vs_base(sim_pp, class_names):
    """For each novel class, compute proto_margin = 1 - sim(novel, nearest_base).

    proto_margin is the prototype-space gap between the novel class and its
    nearest base-class neighbour. Small margin = fragile classification:
    a query embedding only needs to lean slightly toward the base class to fail.
    This does NOT guarantee failure — a well-aligned query embedding can still
    produce correct argmax — but it tells you how much tolerance you have.
    """
    print("\n=== Novel → Nearest Base Class (proto_margin = 1 - nearest_base_sim) ===")
    print("  proto_margin measures classification fragility in prototype space.")
    print("  Small margin = narrow gap; embedding must be very well-aligned to classify correctly.")
    print(f"  FRAGILE:  proto_margin < {FRAGILE_MARGIN_THRESH:.2f}")
    print(f"  MODERATE: proto_margin < {MODERATE_MARGIN_THRESH:.2f}")

    rows = []
    results = {}

    for nid in NOVEL_IDS:
        name = class_names[nid]
        base_sims = [(sim_pp[nid, bid].item(), class_names[bid], bid) for bid in BASE_IDS]
        base_sims.sort(reverse=True)
        top_sim, top_name, _ = base_sims[0]
        margin = 1.0 - top_sim

        rows.append((margin, nid, name, top_sim, top_name, base_sims))

    # Sort by smallest margin first (most fragile)
    rows.sort(key=lambda r: r[0])

    for margin, nid, name, top_sim, top_name, base_sims in rows:
        if margin < FRAGILE_MARGIN_THRESH:
            flag = "  ← FRAGILE"
            label = "fragile"
        elif margin < MODERATE_MARGIN_THRESH:
            flag = "  ← moderate"
            label = "moderate"
        else:
            flag = "  [OK]"
            label = "ok"

        print(f"  * {name:<20}  nearest_base={top_name:<22} sim={top_sim:.4f}  margin={margin:.4f}{flag}")
        for s, n2, _ in base_sims[1:3]:
            print(f"      (also: {n2:<22} sim={s:.4f}  margin={1-s:.4f})")

        results[name] = {
            "nearest_base":   top_name,
            "nearest_base_sim": round(top_sim, 4),
            "proto_margin":   round(margin, 4),
            "fragility":      label,
        }

    return results


def analyse_novel_vs_negbank(sim_pn, class_names, concepts):
    """For each novel class, find nearest negative bank entry.

    The negative bank appears in the InfoNCE denominator during C1 token loss
    training. If a novel class prototype is close to a negative concept, the
    denominator may create training pressure that distorts the embedding space
    near that class.

    NOTE: The negative bank is NOT used at inference. This analysis identifies
    training geometry risk, not inference classification failure.
    """
    print("\n=== Novel → Nearest Negative Bank Entry (training geometry risk, NOT inference) ===")
    print("  The neg bank only appears in the C1 InfoNCE denominator during training.")
    print("  High sim = risk of distorted embedding geometry for that class during training.")
    print("  This does NOT directly cause argmax to fail at inference.")
    print(f"  HIGH RISK: sim > {NEG_INTERFERE_THRESH:.2f}  |  MODERATE: sim > {NEG_SOFT_THRESH:.2f}")

    results  = {}
    high_risk = []
    moderate  = []

    for nid in NOVEL_IDS:
        name     = class_names[nid]
        neg_sims = sim_pn[nid]
        top3_idx = neg_sims.topk(3).indices.tolist()
        top_sim  = neg_sims[top3_idx[0]].item()
        top_conc = concepts[top3_idx[0]]

        if top_sim > NEG_INTERFERE_THRESH:
            flag = "  ← HIGH training geometry risk"
            high_risk.append(name)
        elif top_sim > NEG_SOFT_THRESH:
            flag = "  ← moderate training geometry risk"
            moderate.append(name)
        else:
            flag = "  [OK]"

        print(f"  * {name:<20} → \"{top_conc}\" sim={top_sim:.4f}{flag}")
        for idx in top3_idx[1:]:
            print(f"      (also: \"{concepts[idx]}\" sim={neg_sims[idx].item():.4f})")

        results[name] = {
            "nearest_neg": top_conc,
            "sim":         round(top_sim, 4),
            "high_risk":   top_sim > NEG_INTERFERE_THRESH,
            "moderate":    top_sim > NEG_SOFT_THRESH,
        }

    return results, high_risk, moderate


def analyse_novel_intra(sim_pp, class_names):
    print("\n=== Novel Intra-class Similarity ===")
    for i, nid1 in enumerate(NOVEL_IDS):
        for nid2 in NOVEL_IDS[i + 1:]:
            s = sim_pp[nid1, nid2].item()
            print(f"  {class_names[nid1]:<20} ↔ {class_names[nid2]:<20}  sim={s:.4f}")


def analyse_base_intra(sim_pp, class_names):
    print("\n=== Base Class High-Similarity Pairs (reference) ===")
    pairs = []
    for i, bid1 in enumerate(BASE_IDS):
        for bid2 in BASE_IDS[i + 1:]:
            pairs.append((sim_pp[bid1, bid2].item(), class_names[bid1], class_names[bid2]))
    pairs.sort(reverse=True)
    for s, n1, n2 in pairs[:5]:
        print(f"  {n1:<20} ↔ {n2:<20}  sim={s:.4f}  margin={1-s:.4f}")


def print_neg_bank_coverage(sim_pn, class_names, concepts):
    print("\n=== All Classes → Max Negative Bank Similarity (training geometry reference) ===")
    for i in range(sim_pn.shape[0]):
        top_sim = sim_pn[i].max().item()
        top_idx = sim_pn[i].argmax().item()
        marker = "* NOVEL" if i in NOVEL_IDS else "  base "
        flag = "  ← WARN" if top_sim > NEG_SOFT_THRESH else ""
        print(f"  [{i:2d}] {marker} {class_names[i]:<22}  max-neg-sim={top_sim:.4f}  (\"{concepts[top_idx]}\"){flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_path",    default="panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")
    parser.add_argument("--neg_bank_path", default="panoptic/clip_prototypes/negative_bank_200.pt")
    parser.add_argument("--out_dir",       default="panoptic/diagnostics")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading prototypes from {args.proto_path}...")
    protos, class_names = load_prototypes(args.proto_path)
    print(f"  {protos.shape[0]} classes, dim {protos.shape[1]}")

    print(f"Loading negative bank from {args.neg_bank_path}...")
    neg, concepts = load_neg_bank(args.neg_bank_path)
    print(f"  {neg.shape[0]} negative concepts, dim {neg.shape[1]}")

    sim_pp = protos @ protos.T   # (16, 16)
    sim_pn = protos @ neg.T      # (16, N_neg)

    print("\n=== Class Index Map ===")
    for i, n in enumerate(class_names):
        marker = "* NOVEL" if i in NOVEL_IDS else "  base "
        print(f"  [{i:2d}] {marker}: {n}")

    print_similarity_matrix(sim_pp, class_names)

    nb_results = analyse_novel_vs_base(sim_pp, class_names)
    nn_results, high_risk, neg_moderate = analyse_novel_vs_negbank(sim_pn, class_names, concepts)

    analyse_novel_intra(sim_pp, class_names)
    analyse_base_intra(sim_pp, class_names)
    print_neg_bank_coverage(sim_pn, class_names, concepts)

    # Verdict
    print("\n" + "=" * 60)
    print("=== VERDICT ===")
    print("=" * 60)

    fragile  = [n for n, r in nb_results.items() if r["fragility"] == "fragile"]
    moderate = [n for n, r in nb_results.items() if r["fragility"] == "moderate"]
    any_issue = bool(fragile or high_risk)

    if fragile:
        print(f"\n  FRAGILE prototype margin (proto_margin < {FRAGILE_MARGIN_THRESH}):")
        for c in fragile:
            r = nb_results[c]
            print(f"    - {c}: margin={r['proto_margin']:.4f} vs {r['nearest_base']}")
        print("  Risk: query embeddings need very precise alignment to classify correctly.")
        print("  Fix: re-engineer text prompts for better separation, or add per-class")
        print("       inference logit offsets to widen the effective margin.")

    if moderate:
        print(f"\n  Moderate prototype margin (proto_margin {FRAGILE_MARGIN_THRESH}–{MODERATE_MARGIN_THRESH}):")
        for c in moderate:
            r = nb_results[c]
            print(f"    - {c}: margin={r['proto_margin']:.4f} vs {r['nearest_base']}")
        print("  Monitor: classification may work if L_O/L_V alignment is strong.")

    if high_risk:
        print(f"\n  High neg-bank training geometry risk (novel-neg sim > {NEG_INTERFERE_THRESH}):")
        for c in high_risk:
            print(f"    - {c}")
        print("  The C1 InfoNCE denominator may distort embedding geometry for these classes")
        print("  during training. This does NOT directly break inference argmax.")
        print("  Fix: rebuild negative bank removing concepts similar to these novel classes.")

    if neg_moderate:
        print(f"\n  Moderate neg-bank risk (novel-neg sim {NEG_SOFT_THRESH}–{NEG_INTERFERE_THRESH}):")
        for c in neg_moderate:
            print(f"    - {c}")

    if not any_issue:
        print("\n  All novel classes pass gates:")
        print(f"    - proto_margin > {FRAGILE_MARGIN_THRESH:.2f} for all novel classes (classification not fragile)")
        print(f"    - No high neg-bank training geometry risk")
        print("  Prototype geometry is NOT the bottleneck for novel PQ=0.")
        print("  Check Script A (camera CLIP coverage) and training dynamics instead.")

    out = {
        "thresholds": {
            "fragile_margin":    FRAGILE_MARGIN_THRESH,
            "moderate_margin":   MODERATE_MARGIN_THRESH,
            "neg_interfere":     NEG_INTERFERE_THRESH,
            "neg_soft":          NEG_SOFT_THRESH,
        },
        "novel_vs_base":    nb_results,
        "novel_vs_negbank": nn_results,
        "verdict": {
            "fragile_classes":       fragile,
            "moderate_classes":      moderate,
            "neg_high_risk_classes": high_risk,
            "neg_moderate_classes":  neg_moderate,
            "any_critical_issue":    any_issue,
        },
        "sim_matrix_16x16": sim_pp.tolist(),
        "class_names":       class_names,
        "novel_ids":         NOVEL_IDS,
        "base_ids":          BASE_IDS,
    }
    out_path = os.path.join(args.out_dir, "prototype_similarity.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
