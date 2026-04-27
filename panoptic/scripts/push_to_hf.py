#!/usr/bin/env python3
"""Push panoptic checkpoints, configs, and docs to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_xxx python panoptic/scripts/push_to_hf.py
    HF_TOKEN=hf_xxx python panoptic/scripts/push_to_hf.py --dry-run
    HF_TOKEN=hf_xxx python panoptic/scripts/push_to_hf.py --repo idifro/LPS_Linear
"""

import argparse
import os
import shutil
import tempfile

# Source -> destination mapping (relative to repo root -> relative in staging)
FILE_MAP = [
    # ── Model design docs ──
    ("docs/model_design/model_arch_v1.md", "README.md"),
    ("docs/model_design/model_arch_v1.md", "docs/model_design/model_arch_v1.md"),
    ("docs/model_design/model_arch_token_v1.md", "docs/model_design/model_arch_token_v1.md"),
    ("docs/open_vocab_ablation.md", "docs/open_vocab_ablation.md"),
    # ── Backbone model ──
    ("models/nuscenes_openseg.pth.tar", "models/nuscenes_openseg.pth.tar"),
    # ── Phase 4: mini checkpoints ──
    ("panoptic/checkpoints/best.pth", "checkpoints_mini/best.pth"),
    ("panoptic/checkpoints/latest.pth", "checkpoints_mini/latest.pth"),
    # ── Phase 4: full nuScenes checkpoints ──
    ("panoptic/checkpoints_full/best.pth", "checkpoints_full/best.pth"),
    ("panoptic/checkpoints_full/latest.pth", "checkpoints_full/latest.pth"),
    ("panoptic/checkpoints_full/linear_0.47PQ.pth", "checkpoints_full/linear_0.47PQ.pth"),
    # ── Phase 5: token v1 (full nuScenes) ──
    ("panoptic/checkpoints_token_full_v1/best.pth", "checkpoints_token_full_v1/best.pth"),
    ("panoptic/checkpoints_token_full_v1/latest.pth", "checkpoints_token_full_v1/latest.pth"),
    ("panoptic/checkpoints_token_full_v1/best_mini_warm_0_41.pth", "checkpoints_token_full_v1/best_mini_warm_0_41.pth"),
    # ── Phase 5: token v2 — contrastive (full nuScenes) ──
    ("panoptic/checkpoints_token_full_v2/best.pth", "checkpoints_token_full_v2/best.pth"),
    ("panoptic/checkpoints_token_full_v2/latest.pth", "checkpoints_token_full_v2/latest.pth"),
    # ── CLIP prototypes ──
    ("panoptic/clip_prototypes/text_prototypes_vitl14_336.pt", "clip_prototypes/text_prototypes_vitl14_336.pt"),
    # ── Saved text embeddings ──
    ("saved_text_embeddings/clip_nuscenes_labels_768.pt", "saved_text_embeddings/clip_nuscenes_labels_768.pt"),
    # ── Configs ──
    ("panoptic/configs/default.yaml", "configs/default.yaml"),
    ("panoptic/configs/full_nuscenes.yaml", "configs/full_nuscenes.yaml"),
    ("panoptic/configs/full_nuscenes_4gpu.yaml", "configs/full_nuscenes_4gpu.yaml"),
    ("panoptic/configs/full_nuscenes_token.yaml", "configs/full_nuscenes_token.yaml"),
    ("panoptic/configs/full_nuscenes_token_v2.yaml", "configs/full_nuscenes_token_v2.yaml"),
    ("panoptic/configs/mini_token.yaml", "configs/mini_token.yaml"),
]

REPO_ROOT = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene"


def main():
    parser = argparse.ArgumentParser(description="Push model artifacts to HuggingFace Hub")
    parser.add_argument("--repo", default="idifro/LPS_Linear", help="HuggingFace repo ID")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    # Auth
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: Set HF_TOKEN environment variable (get from https://huggingface.co/settings/tokens)")
        return 1

    # Stage files
    staging_dir = tempfile.mkdtemp(prefix="hf_upload_")
    print(f"Staging directory: {staging_dir}")

    copied = []
    skipped = []
    for src_rel, dst_rel in FILE_MAP:
        src = os.path.join(REPO_ROOT, src_rel)
        dst = os.path.join(staging_dir, dst_rel)

        if not os.path.exists(src):
            skipped.append(src_rel)
            print(f"  SKIP (missing): {src_rel}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(src) / (1024 * 1024)
        copied.append((src_rel, dst_rel, size_mb))
        print(f"  COPY: {src_rel} -> {dst_rel} ({size_mb:.1f} MB)")

    print(f"\nTotal: {len(copied)} files copied, {len(skipped)} skipped")

    if skipped:
        print(f"Skipped files: {skipped}")
        print("(These may still be training — re-run later to push them)")

    if args.dry_run:
        print("\n--dry-run: No upload performed.")
        shutil.rmtree(staging_dir)
        return 0

    if not copied:
        print("Nothing to upload.")
        shutil.rmtree(staging_dir)
        return 1

    # Upload
    print(f"\nUploading to https://huggingface.co/{args.repo} ...")
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=staging_dir,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload panoptic checkpoints, configs, and docs",
    )

    print(f"\nDone! View at: https://huggingface.co/{args.repo}")

    # Cleanup
    shutil.rmtree(staging_dir)
    return 0


if __name__ == "__main__":
    exit(main())
