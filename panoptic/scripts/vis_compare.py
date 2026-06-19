#!/usr/bin/env python3
"""
Visualize infer_compare_out results as PNG files.
No display/X11 needed — saves to vis_out/.
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

PALETTE = np.random.RandomState(42).randint(50, 255, (300, 3)).astype(np.uint8)

def load_sample(infer_dir, tag, idx):
    d = Path(infer_dir) / tag
    files = sorted(d.glob(f"{idx:04d}_*_coords.npy"))
    if not files:
        return None
    base = str(files[0]).replace("_coords.npy", "")
    coords   = np.load(f"{base}_coords.npy")   # (V,3)
    sem_pred = np.load(f"{base}_sem_pred.npy")  # (V,)
    inst_pred= np.load(f"{base}_inst_pred.npy") # (V,)
    bb_feats = np.load(f"{base}_bb_feats.npy")  # (V,768)
    return coords, sem_pred, inst_pred, bb_feats

def bev_image(coords, labels, title, ax, point_size=0.3):
    """Bird's eye view (X-Y plane) coloured by label."""
    cols = PALETTE[labels % len(PALETTE)] / 255.0
    ax.scatter(coords[:, 0], coords[:, 1], c=cols,
               s=point_size, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dir",   default="infer_compare_out")
    parser.add_argument("--out_dir",     default="vis_out")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cos_files = sorted(Path(args.infer_dir).glob("*_feat_cos_sim.npy"))
    if not cos_files:
        print("No inference outputs found. Run infer_compare.py first.")
        return

    for i, cos_file in enumerate(cos_files):
        if args.max_samples and i >= args.max_samples:
            break

        idx = int(cos_file.name[:4])
        cos_sim = np.load(cos_file)   # (V,)

        a = load_sample(args.infer_dir, "all_bb",   idx)
        b = load_sample(args.infer_dir, "dec_only", idx)
        if a is None or b is None:
            continue

        coords_a, sem_a, inst_a, feats_a = a
        coords_b, sem_b, inst_b, feats_b = b

        # semantic agreement mask
        agree = (sem_a == sem_b).astype(np.uint8)  # 1=agree, 0=disagree

        # feature norm per point
        feat_norm_a = np.linalg.norm(feats_a, axis=1)
        feat_norm_b = np.linalg.norm(feats_b, axis=1)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Sample {idx}  |  feat_cos={cos_sim.mean():.4f}  "
                     f"sem_agree={agree.mean()*100:.1f}%", fontsize=10)

        # row 0: AllBB run
        bev_image(coords_a, sem_a,  "AllBB — Semantic pred",   axes[0, 0])
        bev_image(coords_a, inst_a, "AllBB — Instance pred",   axes[0, 1])
        axes[0, 2].scatter(coords_a[:, 0], coords_a[:, 1],
                           c=feat_norm_a, cmap="plasma",
                           s=0.3, linewidths=0)
        axes[0, 2].set_title("AllBB — BB feat norm", fontsize=8)
        axes[0, 2].set_aspect("equal"); axes[0, 2].axis("off")
        axes[0, 3].hist(feat_norm_a, bins=80, color="steelblue", alpha=0.7)
        axes[0, 3].set_title("AllBB — feat norm dist", fontsize=8)

        # row 1: DecOnly run
        bev_image(coords_b, sem_b,  "DecOnly — Semantic pred", axes[1, 0])
        bev_image(coords_b, inst_b, "DecOnly — Instance pred", axes[1, 1])
        axes[1, 2].scatter(coords_b[:, 0], coords_b[:, 1],
                           c=feat_norm_b, cmap="plasma",
                           s=0.3, linewidths=0)
        axes[1, 2].set_title("DecOnly — BB feat norm", fontsize=8)
        axes[1, 2].set_aspect("equal"); axes[1, 2].axis("off")
        axes[1, 3].hist(feat_norm_b, bins=80, color="darkorange", alpha=0.7)
        axes[1, 3].set_title("DecOnly — feat norm dist", fontsize=8)

        # agreement overlay on top of AllBB coords
        # overlay disagreement points in red
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle(f"Sample {idx} — Semantic agreement", fontsize=10)

        bev_image(coords_a, sem_a, "AllBB sem", ax2[0])
        bev_image(coords_b, sem_b, "DecOnly sem", ax2[1])

        cols_agree = np.where(agree[:, None], [[0.2, 0.8, 0.2]], [[0.9, 0.1, 0.1]])
        ax2[2].scatter(coords_a[:, 0], coords_a[:, 1],
                       c=cols_agree, s=0.3, linewidths=0)
        ax2[2].set_title(f"Agreement (green) / Disagree (red)\n"
                         f"{agree.mean()*100:.1f}% agree", fontsize=8)
        ax2[2].set_aspect("equal"); ax2[2].axis("off")

        # save
        out1 = os.path.join(args.out_dir, f"{idx:04d}_compare.png")
        out2 = os.path.join(args.out_dir, f"{idx:04d}_agreement.png")
        fig.savefig(out1, dpi=150, bbox_inches="tight")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close("all")

        print(f"[{i+1}] saved {out1}")
        print(f"      saved {out2}")

    print(f"\n[✓] PNGs in {args.out_dir}/")

if __name__ == "__main__":
    main()