"""verify_neg_bank.py — GATE 1 verification for negative_bank_v2.pt.

Checks:
  1. Zero entries with sim > class_sim_thresh to any class proto
  2. Bank size >= 100
  3. Diversity: mean pairwise sim < 0.65
  4. Per-novel-class max sim (novel classes should not be polluted)

Pass criteria (all must pass to proceed to Phase 2):
  GATE 1a: zero entries with sim > 0.75 to any class proto
  GATE 1b: bank size >= 100
  GATE 1c: novel-class max sim < 0.80  (bus, motorcycle, pedestrian, vegetation)

Usage:
    python panoptic/scripts/verify_neg_bank.py \
        --neg_bank panoptic/clip_prototypes/negative_bank_v2.pt \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
"""
import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

NOVEL_IDS = [2, 5, 6, 15]
NOVEL_NAMES = {2: "bus", 5: "motorcycle", 6: "pedestrian", 15: "vegetation"}

CLASS_SIM_THRESH   = 0.75
MIN_BANK_SIZE      = 100
NOVEL_SIM_THRESH   = 0.80   # max allowed sim of any neg-bank entry to novel proto
DIVERSITY_MAX_MEAN = 0.65   # mean pairwise sim gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg_bank",   default="panoptic/clip_prototypes/negative_bank_v2.pt")
    parser.add_argument("--proto_path", default="panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print(f"Loading negative bank: {args.neg_bank}")
    nb = torch.load(args.neg_bank, map_location="cpu")
    embs = nb["embeddings"].float()          # (N, 768)
    concepts = nb.get("concepts", [])
    embs = F.normalize(embs, dim=-1)
    N = embs.shape[0]
    print(f"  Bank size: {N}")

    print(f"\nLoading prototypes: {args.proto_path}")
    pd = torch.load(args.proto_path, map_location="cpu")
    protos = pd["embeddings"].float() if isinstance(pd, dict) else pd.float()
    class_names = pd.get("class_names", [str(i) for i in range(16)]) if isinstance(pd, dict) else [str(i) for i in range(16)]
    protos = F.normalize(protos, dim=-1)     # (16, 768)

    # ── Gate 1a: no entry > CLASS_SIM_THRESH ──────────────────────────────────
    sim_to_class = embs @ protos.T           # (N, 16)
    max_sim, nearest_cls = sim_to_class.max(dim=1)

    violations = (max_sim > CLASS_SIM_THRESH).nonzero(as_tuple=True)[0]
    gate_1a = len(violations) == 0

    print(f"\nGATE 1a — zero entries with sim>{CLASS_SIM_THRESH} to any class proto:")
    if gate_1a:
        print(f"  PASS ✓  (max sim = {max_sim.max():.4f})")
    else:
        print(f"  FAIL ✗  ({len(violations)} violations, max sim = {max_sim.max():.4f})")
        for vi in violations[:10]:
            cn = class_names[nearest_cls[vi]] if nearest_cls[vi] < len(class_names) else str(nearest_cls[vi].item())
            print(f"    [{vi:3d}] sim={max_sim[vi]:.4f} → {cn:<20}  '{concepts[vi] if vi < len(concepts) else '?'}'")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")

    # ── Gate 1b: bank size ─────────────────────────────────────────────────────
    gate_1b = N >= MIN_BANK_SIZE
    print(f"\nGATE 1b — bank size >= {MIN_BANK_SIZE}:")
    if gate_1b:
        print(f"  PASS ✓  (N = {N})")
    else:
        print(f"  FAIL ✗  (N = {N} < {MIN_BANK_SIZE})")

    # ── Gate 1c: novel-class max sim ─────────────────────────────────────────
    print(f"\nGATE 1c — per-novel-class max sim < {NOVEL_SIM_THRESH}:")
    gate_1c = True
    for nid in NOVEL_IDS:
        if nid >= protos.shape[0]:
            continue
        sim_novel = (embs @ protos[nid]).max().item()
        cname = NOVEL_NAMES.get(nid, str(nid))
        ok = sim_novel < NOVEL_SIM_THRESH
        flag = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{nid:2d}] {cname:<15}: max_sim = {sim_novel:.4f}  {flag}")
        if not ok:
            gate_1c = False
            # Print the worst offenders
            novel_sims = embs @ protos[nid]
            topk_idx = novel_sims.argsort(descending=True)[:3]
            for ti in topk_idx:
                print(f"       → sim={novel_sims[ti]:.4f}  '{concepts[ti] if ti < len(concepts) else '?'}'")

    # ── Informational: diversity ───────────────────────────────────────────────
    print(f"\nINFO — Diversity (mean pairwise sim, lower is more diverse):")
    if N <= 500:
        sim_matrix = embs @ embs.T          # (N, N)
        mask = ~torch.eye(N, dtype=torch.bool)
        mean_pairwise = sim_matrix[mask].mean().item()
        max_pairwise  = sim_matrix[mask].max().item()
        print(f"  Mean pairwise sim: {mean_pairwise:.4f}  (gate: < {DIVERSITY_MAX_MEAN})")
        print(f"  Max  pairwise sim: {max_pairwise:.4f}")
        div_ok = mean_pairwise < DIVERSITY_MAX_MEAN
        print(f"  {'PASS ✓' if div_ok else 'WARN (non-blocking) — mean pairwise sim high'}")
    else:
        print(f"  Skipping (N={N} > 500)")

    # ── Informational: nearest-class distribution ─────────────────────────────
    if args.verbose:
        print(f"\nINFO — Nearest-class distribution:")
        for cls_i, cname in enumerate(class_names):
            count = (nearest_cls == cls_i).sum().item()
            if count > 0:
                print(f"  [{cls_i:2d}] {cname:<25}: {count} entries  (max_sim={sim_to_class[:, cls_i].max():.4f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = gate_1a and gate_1b and gate_1c
    print("\n" + "=" * 60)
    print("GATE SUMMARY:")
    print(f"  1a (zero class violations):   {'PASS ✓' if gate_1a else 'FAIL ✗'}")
    print(f"  1b (size >= {MIN_BANK_SIZE}):              {'PASS ✓' if gate_1b else 'FAIL ✗'}")
    print(f"  1c (novel max sim < {NOVEL_SIM_THRESH}):   {'PASS ✓' if gate_1c else 'FAIL ✗'}")
    print()
    if all_pass:
        print("  → ALL GATES PASS. Proceed to Phase 2 (generate_prototypes_v2.py).")
    else:
        print("  → GATES FAILED. Fix issues before proceeding.")
        if not gate_1a:
            print("    Fix: lower class_sim_thresh or remove flagged concepts from candidate list.")
        if not gate_1b:
            print("    Fix: add more candidate concepts or raise diversity_thresh slightly.")
        if not gate_1c:
            print("    Fix: add specific exclusions for concepts near novel classes.")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
