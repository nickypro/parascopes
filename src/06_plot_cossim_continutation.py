import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import torch
torch.set_grad_enabled(False)

try:
    from utils_load_data import BASE_DIR
except Exception:
    # Fallback: assume the project root is 2 levels up from this file
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

BASE_DIR="/workspace/nicky/parascopes/src/"

# CONFIG
EMBED_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 512  # pairs per embed batch


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_cosine_similarity(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    dot_products = np.sum(reference_embeddings * candidate_embeddings, axis=1)
    ref_norm = np.linalg.norm(reference_embeddings, axis=1)
    cand_norm = np.linalg.norm(candidate_embeddings, axis=1)
    denom = (ref_norm * cand_norm) + 1e-8
    return dot_products / denom


def load_original_generated_pairs(json_path: str) -> List[Tuple[str, str]]:
    """
    Load (original, generated) pairs from a JSON file.
    The file is a single JSON object with an 'outputs' list containing dicts
    with 'original' and 'generated' keys.
    Skips pairs where original == "" (or whitespace-only).
    """
    pairs = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    outputs = data.get("outputs", [])
    if not isinstance(outputs, list):
        return pairs
    
    for item in outputs:
        if not isinstance(item, dict):
            continue
        orig = item.get("original", None)
        gen = item.get("generated", None)
        if not isinstance(orig, str) or orig == "" or orig.strip() == "":
            # Skip if original is empty (as requested)
            continue
        if not isinstance(gen, str):
            # Skip malformed generated entries
            continue
        pairs.append((orig, gen))
    
    return pairs


def cosine_sims_from_json(json_path: str, model: SentenceTransformer, batch_size: int = BATCH_SIZE) -> List[float]:
    """
    Load pairs from a JSON file and compute cosine similarity between
    (original, generated) pairs in batches.
    """
    model_name = os.path.basename(os.path.dirname(json_path))
    print(f"Loading pairs from {model_name}...")
    pairs = load_original_generated_pairs(json_path)
    
    if not pairs:
        print(f"  No valid pairs found in {model_name}")
        return []
    
    print(f"  Found {len(pairs)} pairs, computing embeddings...")
    
    sims: List[float] = []
    
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"Embedding {model_name}"):
        batch = pairs[i:i + batch_size]
        orig_batch = [p[0] for p in batch]
        gen_batch = [p[1] for p in batch]
        
        ref_emb = model.encode(orig_batch, convert_to_numpy=True, show_progress_bar=False)
        cand_emb = model.encode(gen_batch, convert_to_numpy=True, show_progress_bar=False)
        cossims = compute_cosine_similarity(ref_emb, cand_emb)
        sims.extend(cossims.tolist())

    print(f"  Finished {model_name}: {len(sims)} similarities computed")
    return sims


def plot_cosine_similarity_violin(
    df: pd.DataFrame,
    output_image_path: str,
    colour_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    plt.figure(figsize=(11, 6))

    labels = list(df["Model"].unique())
    if colour_map is None:
        base_palette = sns.color_palette("husl", n_colors=len(labels))
        palette = {label: base_palette[i] for i, label in enumerate(labels)}
    else:
        palette = {label: colour_map.get(label, (0.6, 0.6, 0.6)) for label in labels}

    sns.violinplot(
        data=df,
        x="Model",
        y="Cosine Similarity",
        hue="Model",
        palette=palette,
        scale="width",
        cut=0,
        legend=False,
    )
    plt.title("Cosine Similarities: original vs generated (continuation)")
    plt.ylim(top=1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    ensure_dir(output_image_path)
    plt.savefig(output_image_path, dpi=300)
    print(f"\nViolin plot saved to: {output_image_path}")
    plt.show()

    summary = df.groupby("Model")["Cosine Similarity"].agg(["count", "mean", "std", "sem"]).sort_values("mean", ascending=False)
    print("\nSummary by model:")
    print(summary)


if __name__ == "__main__":
    # Initialize embedding model
    print(f"Initializing SentenceTransformer: {EMBED_MODEL_REPO}")
    try:
        model = SentenceTransformer(EMBED_MODEL_REPO, model_kwargs={"dtype": "bfloat16"})
    except Exception:
        # Fallback if bfloat16 unsupported in runtime
        model = SentenceTransformer(EMBED_MODEL_REPO)

    # Files to load (as listed)
    files = {
        "gemma 270m": f"{BASE_DIR}/comparison_texts/gemma-270m/transferred_activation_output.jsonl",
        "gemma 1b": f"{BASE_DIR}/comparison_texts/gemma-1b/transferred_activation_output.jsonl",
        "gemma 4b": f"{BASE_DIR}/comparison_texts/gemma-4b/transferred_activation_output.jsonl",
        "gemma 12b": f"{BASE_DIR}/comparison_texts/gemma-12b/transferred_activation_output.jsonl",
        "gemma 27b": f"{BASE_DIR}/comparison_texts/gemma-27b/transferred_activation_output.jsonl",
    }

    # Compute cosine similarities for each model family
    all_sims: List[float] = []
    all_labels: List[str] = []

    for label, path in files.items():
        if not os.path.isfile(path):
            print(f"Warning: missing file for {label}: {path}")
            continue
        sims = cosine_sims_from_json(path, model, batch_size=BATCH_SIZE)
        if not sims:
            print(f"Warning: no valid pairs for {label}")
            continue
        all_sims.extend(sims)
        all_labels.extend([label] * len(sims))

    if not all_sims:
        raise SystemExit("No cosine similarities computed. Check input files and formats.")

    df_plot = pd.DataFrame(
        {
            "Cosine Similarity": all_sims,
            "Model": all_labels,
        }
    )

    # Save CSV
    csv_out = f"{BASE_DIR}/cached_results/cossim_contuniation.csv"
    ensure_dir(csv_out)
    df_plot.to_csv(csv_out, index=False)
    print(f"Cosine similarity details saved to: {csv_out}")

    # Plot
    fig_out = f"{BASE_DIR}/figures/cossim-contuniation.png"
    plot_cosine_similarity_violin(df_plot, fig_out, colour_map=None)