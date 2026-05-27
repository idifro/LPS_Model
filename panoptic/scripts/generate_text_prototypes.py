"""Generate CLIP text prototypes for nuScenes 16-class panoptic segmentation.

Uses OpenAI CLIP ViT-L/14@336px (same model the backbone was distilled from).
Prompt template: "a {label} in a scene" (matches util/util.py:extract_text_feature).

Output: panoptic/clip_prototypes/text_prototypes_vitl14_336.pt
  - embeddings: (16, 768) L2-normalized text features (one per class, 0-indexed)
  - class_names: list of 16 class name strings
  - prompt_template: the template used
  - model_name: CLIP model name

Usage:
    python panoptic/scripts/generate_text_prototypes.py [--output PATH] [--device cuda]
"""
import argparse
import sys
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

# nuScenes 16-class names (0-indexed, matching dataset convention minus the 1-index offset)
NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

PROMPT_TEMPLATE = "a {} in a scene"


def main():
    """Generate and save L2-normalized CLIP text prototypes for nuScenes 16-class panoptic segmentation.

    Produces a reusable prototype file that maps each semantic class to a 768-d CLIP text
    embedding, used downstream for open-vocabulary classification and loss computation.

    Steps:
        1. Parses CLI arguments for output path, device, and CLIP model variant.
        2. Loads the specified CLIP model (default ViT-L/14@336px) in eval mode.
        3. Formats each of the 16 class names with the prompt template "a {label} in a scene".
        4. Tokenizes prompts, encodes with CLIP text encoder, and L2-normalizes embeddings.
        5. Saves a dictionary with embeddings (16, 768), class_names, prompt_template,
           and model_name to the output .pt file.

    Returns:
        None: Saves the prototype file to disk and prints confirmation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="panoptic/clip_prototypes/text_prototypes_vitl14_336.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="ViT-L/14@336px")
    args = parser.parse_args()

    import clip
    print(f"Loading CLIP {args.model_name}...", flush=True)
    model, _ = clip.load(args.model_name, device=args.device, jit=False)
    model.eval()

    prompts = [PROMPT_TEMPLATE.format(name) for name in NUSCENES_CLASS_NAMES]
    print(f"Generating embeddings for {len(prompts)} classes:", flush=True)
    for i, p in enumerate(prompts):
        print(f"  [{i:2d}] {p}", flush=True)

    tokens = clip.tokenize(prompts).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)  # (16, 768)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.float().cpu()
    print(f"\nEmbeddings shape: {text_features.shape}", flush=True)
    print(f"Norm check (should be ~1.0): {text_features.norm(dim=-1)}", flush=True)

    output = {
        "embeddings": text_features,
        "class_names": NUSCENES_CLASS_NAMES,
        "prompt_template": PROMPT_TEMPLATE,
        "model_name": args.model_name,
    }
    torch.save(output, args.output)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
