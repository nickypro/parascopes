"""
Normalization utilities for residual streams and embeddings.
"""

import torch
import einops
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from utils_load_data import load_residuals, load_embeds

logger = logging.getLogger(__name__)

SAMPLES_PER_CHUNK = 1000  # invariant


@dataclass
class WelfordStats:
    mean: torch.Tensor = None
    m2: torch.Tensor = None
    count: int = 0
    def update(self, new_data: torch.Tensor):
        if self.mean is None:
            self.mean = torch.zeros_like(new_data[0])
            self.m2 = torch.zeros_like(new_data[0])
            self.count = 0
        for x in new_data:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
    @property
    def std(self):
        denom = max(self.count - 1, 1)
        return torch.sqrt(self.m2 / denom + 1e-6)


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean


def compute_normalizers(
    norm_chunk_ids: List[int],
    hf_repo_residuals: str,
    hf_repo_embeds: str,
    local_residuals_dir: Optional[str],
    local_embeds_dir: Optional[str],
    dtype: torch.dtype = torch.float32,
) -> Tuple[Normalizer, Normalizer]:
    """
    Compute mean/std using Welford over a small subset of chunks.
    Skip any chunk where residuals or embeddings length != 1000.
    """
    print("\nComputing normalization stats (subset)...")
    logger.info("Computing normalization stats (subset)...")

    res_stats = WelfordStats()
    embed_stats = WelfordStats()
    res_reshape = "layer para dim -> para layer dim"

    used, skipped = 0, []

    for chunk_id in tqdm(norm_chunk_ids, desc="Stats chunks"):
        try:
            res_list = load_residuals(
                chunk_id, hf_repo_residuals, local_residuals_dir
            )
            embeds = load_embeds(
                chunk_id, hf_repo_embeds,local_embeds_dir
            )

            n_res = len(res_list)
            n_emb = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)

            if n_res != SAMPLES_PER_CHUNK or n_emb != SAMPLES_PER_CHUNK:
                logger.warning(f"[norm] skip chunk {chunk_id:03d}: res={n_res}, emb={n_emb} (expected 1000).")
                skipped.append(chunk_id); continue

            for res, embed in zip(res_list, embeds):
                res_all = res["res"].to(dtype=dtype)                  # [n_layers, n_para, d_model]
                first_para = res_all[:, :1, :]                        # first paragraph only
                res_tensor = einops.rearrange(first_para, res_reshape)  # [1, n_layers, d_model]

                res_stats.update(res_tensor)
                embed_stats.update(embed.unsqueeze(0))  # embed_stats.mean and embed_stats.std have shape [d_sonar]

            used += 1

        except Exception as e:
            logger.warning(f"[norm] failed chunk {chunk_id:03d}: {e}. Skipping.")
            skipped.append(chunk_id); continue

    if used == 0:
        raise RuntimeError("No valid chunks for normalization (after skipping mismatches).")

    logger.info(f"[norm] used {used}/{len(norm_chunk_ids)} chunks; skipped={skipped}")

    res_normalizer = Normalizer(res_stats.mean, res_stats.std)
    embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std)
    return res_normalizer, embed_normalizer


def save_normalizers(res_normalizer: Normalizer, embed_normalizer: Normalizer, out_dir: Path):
    """Save residual and embedding normalizers to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "res_normalizer.pt"
    emb_path = out_dir / "embed_normalizer.pt"
    torch.save({"mean": res_normalizer.mean, "std": res_normalizer.std}, res_path)
    torch.save({"mean": embed_normalizer.mean, "std": embed_normalizer.std}, emb_path)
    return str(res_path), str(emb_path)
