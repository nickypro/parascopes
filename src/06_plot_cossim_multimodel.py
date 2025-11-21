import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils_load_data import BASE_DIR

import torch
torch.set_grad_enabled(False)

# CONFIG
# EMBED_MODEL_REPO = "all-mpnet-base-v2"
EMBED_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B"


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_texts(file_path: str) -> List[str]:
    """
    Load a JSON file that contains texts. Supports raw list or {'outputs': list}.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "outputs" in data:
        texts = data["outputs"]
    else:
        texts = data
    if not isinstance(texts, list):
        raise ValueError(f"Expected a list of texts in {file_path}, got {type(texts)}")
    return texts


def compute_cosine_similarity(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity between rows of reference and candidate embeddings.
    Shapes: (N, D) vs (N, D) -> (N,)
    """
    dot_products = np.sum(reference_embeddings * candidate_embeddings, axis=1)
    ref_norm = np.linalg.norm(reference_embeddings, axis=1)
    cand_norm = np.linalg.norm(candidate_embeddings, axis=1)
    denom = (ref_norm * cand_norm) + 1e-8
    return dot_products / denom


def get_cosine_similarity_across_pairs(
    compare_pairs: Dict[str, Tuple[str, str]],
    embed_model_repo: str = EMBED_MODEL_REPO,
) -> pd.DataFrame:
    """
    Compute cosine similarities for multiple (candidate, reference) text file pairs.

    compare_pairs maps a human-friendly label to a tuple:
      label -> (candidate_json_path, reference_json_path)
    """
    print(f"Initializing SentenceTransformer: {embed_model_repo}")
    model = SentenceTransformer(embed_model_repo, model_kwargs={"dtype": "bfloat16"})

    texts_cache: Dict[str, List[str]] = {}
    embeddings_cache: Dict[str, np.ndarray] = {}

    def get_embeddings(json_path: str) -> np.ndarray:
        if json_path in embeddings_cache:
            return embeddings_cache[json_path]
        if json_path not in texts_cache:
            texts_cache[json_path] = load_texts(json_path)
        embeddings = model.encode(texts_cache[json_path], convert_to_numpy=True)
        embeddings_cache[json_path] = embeddings
        return embeddings

    all_similarities: List[float] = []
    all_labels: List[str] = []

    for label, (candidate_path, reference_path) in (pbar := tqdm(compare_pairs.items(), total=len(compare_pairs))):
        pbar.set_description(f"Comparing {label}")
        cand_emb = get_embeddings(candidate_path)
        ref_emb = get_embeddings(reference_path)
        if cand_emb.shape[0] != ref_emb.shape[0]:
            min_len = min(cand_emb.shape[0], ref_emb.shape[0])
            print(f"Warning: '{label}' count mismatch ({cand_emb.shape[0]} vs {ref_emb.shape[0]}). Truncating to {min_len}.")
            cand_emb = cand_emb[:min_len]
            ref_emb = ref_emb[:min_len]
        if cand_emb.shape[0] == 0:
            print(f"Warning: '{label}' has zero texts after truncation; skipping.")
            continue

        cossims = compute_cosine_similarity(ref_emb, cand_emb)
        all_similarities.extend(cossims.tolist())
        all_labels.extend([label] * cossims.shape[0])

    df = pd.DataFrame(
        {
        "Cosine Similarity": all_similarities,
            "Comparison": all_labels,
        }
    )
    return df


def plot_cosine_similarity_violin(
    df: pd.DataFrame,
    output_image_path: str,
    colour_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    """
    Plot a violin plot of cosine similarities by comparison label.
    """
    plt.figure(figsize=(11, 6))

    labels = list(df["Comparison"].unique())
    if colour_map is None:
        base_palette = sns.color_palette("husl", n_colors=len(labels))
        palette = {label: base_palette[i] for i, label in enumerate(labels)}
    else:
        palette = {label: colour_map.get(label, (0.6, 0.6, 0.6)) for label in labels}
    
    sns.violinplot(
        data=df,
        x="Comparison",
        y="Cosine Similarity",
        hue="Comparison",
        palette=palette,
        scale="width",
        cut=0,
        legend=False,
    )
    plt.title("Cosine Similarities (candidate vs reference)")
    plt.ylim(top=1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    ensure_dir(output_image_path)
    plt.savefig(output_image_path, dpi=300)
    print(f"\nViolin plot saved to: {output_image_path}")
    plt.show()
    summary = df.groupby("Comparison")["Cosine Similarity"].agg(["mean", "std", "sem"]).sort_values("mean", ascending=False)
    print("\nSummary (mean ± std ± stderr) by comparison:")
    print(summary)


if __name__ == "__main__":
    # Compare all Gemma families against their own references
    gemma270_dir = f"{BASE_DIR}/comparison_texts/gemma-270m"
    gemma1b_dir = f"{BASE_DIR}/comparison_texts/gemma-1b"
    gemma4b_dir = f"{BASE_DIR}/comparison_texts/gemma-4b"
    gemma12b_dir = f"{BASE_DIR}/comparison_texts/gemma-12b"
    gemma27b_dir = f"{BASE_DIR}/comparison_texts/gemma-27b"

    compare_pairs = {
        # 270m
        "gemma 270m v1": (f"{gemma270_dir}/linear_decoded_texts_v1.json", f"{gemma270_dir}/original_texts.json"),
        # 1b
        "gemma 1b v1": (f"{gemma1b_dir}/linear_decoded_texts_v1.json", f"{gemma1b_dir}/original_texts.json"),
        # 4b
        "gemma 4b v1": (f"{gemma4b_dir}/linear_decoded_texts_v1.json", f"{gemma4b_dir}/original_texts.json"),
        "gemma 4b v2": (f"{gemma4b_dir}/linear_decoded_texts_v2.json", f"{gemma4b_dir}/original_texts.json"),
        # 12b
        "gemma 12b v1": (f"{gemma12b_dir}/linear_decoded_texts_v1.json", f"{gemma12b_dir}/original_texts.json"),
        "gemma 12b v2": (f"{gemma12b_dir}/linear_decoded_texts_v2.json", f"{gemma12b_dir}/original_texts.json"),
        # 27b
        "gemma 27b v1": (f"{gemma27b_dir}/linear_decoded_texts_v1.json", f"{gemma27b_dir}/original_texts.json"),
    }

    # Compute similarities
    df_plot = get_cosine_similarity_across_pairs(compare_pairs, embed_model_repo=EMBED_MODEL_REPO)

    # Save CSV
    csv_out = f"{BASE_DIR}/cached_results/cossim_multimodel.csv"
    ensure_dir(csv_out)
    df_plot.to_csv(csv_out, index=False)
    print(f"Cosine similarity details saved to: {csv_out}")

    # Plot
    fig_out = f"{BASE_DIR}/figures/cossim-multimodel.png"
    plot_cosine_similarity_violin(df_plot, fig_out, colour_map=None)

