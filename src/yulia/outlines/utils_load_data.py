"""
Utilities for loading embeddings, residuals, and outlines from HuggingFace or local storage.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Import from both config files since this is a shared utility
try:
    from config import HF_OUTLINES_REPO
except ImportError:
    from yulia.outlines.config import HF_OUTLINES_REPO

try:
    from config_probe import HF_EMBEDDINGS_REPO
except ImportError:
    from yulia.outlines.config_probe import HF_EMBEDDINGS_REPO


def _load_torch_file(file_path: Path):
    return torch.load(file_path, map_location="cpu", weights_only=False)


def _load_parquet_file(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)


def _download_hf_torch(repo_id: str, filename: str):
    hf_token = os.environ.get("HF_TOKEN")
    fpath = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
    )
    return _load_torch_file(fpath)


def _download_hf_parquet(repo_id: str, filename: str) -> pd.DataFrame:
    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        repo_id,
        data_files=filename,
        split="train",
        token=hf_token,
    )
    return dataset.to_pandas()


def load_embeds(
    chunk_id: int,
    local_dir: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    cast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Load embeddings for a given chunk.
    """
    filename = f"outlines_{chunk_id:03d}.pt"
    errors = []

    # Try local first if provided
    if local_dir:
        try:
            local_path = Path(local_dir) / filename
            if local_path.exists():
                embeds = _load_torch_file(local_path)
                if cast_dtype is not None:
                    embeds = embeds.to(dtype=cast_dtype)
                print(f"Loaded embeddings from: {local_path}")
                return embeds
        except Exception as e:
            errors.append(f"Local loading failed: {e}")

    # Try HF if provided
    if hf_repo_id or not local_dir:
        if hf_repo_id is None:
            hf_repo_id = HF_EMBEDDINGS_REPO
        try:
            embeds = _download_hf_torch(hf_repo_id, filename)
            if cast_dtype is not None:
                embeds = embeds.to(dtype=cast_dtype)
            print(f"Downloaded embeddings from HF: {hf_repo_id}/{filename}")
            return embeds
        except Exception as e:
            errors.append(f"HuggingFace loading failed: {e}")

    raise RuntimeError(f"Failed to load embeddings {filename}:\n" + "\n".join(errors))


def load_residuals(
    chunk_id: int,
    local_dir: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
) -> Union[list, torch.Tensor]:
    """
    Load residual dumps for a given chunk.
    """
    filename = f"res_data_{chunk_id:03d}.pt"
    errors = []

    # Try local first if provided
    if local_dir:
        try:
            local_path = Path(local_dir) / filename
            if local_path.exists():
                residuals = _load_torch_file(local_path)
                print(f"Loaded residuals from: {local_path}")
                return residuals
            else:
                errors.append(f"Local file not found: {local_path}")
        except Exception as e:
            errors.append(f"Local loading failed: {e}")

    # Try HF if provided
    if hf_repo_id or not local_dir:
        if hf_repo_id is not None:
            try:
                residuals = _download_hf_torch(hf_repo_id, filename)
                print(f"Downloaded residuals from HF: {hf_repo_id}/{filename}")
                return residuals
            except Exception as e:
                errors.append(f"HuggingFace loading failed: {e}")

    raise RuntimeError(f"Failed to load residuals {filename}:\n" + "\n".join(errors))


def load_outlines(
    chunk_id: int,
    local_dir: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    version: str = "v0.0",
) -> pd.DataFrame:
    """
    Load outlines metadata (parquet) for a given chunk.
    """
    filename = f"outlines_{chunk_id:03d}.parquet"
    errors = []

    # Try local first if provided
    if local_dir:
        try:
            local_path = Path(local_dir) / filename
            if local_path.exists():
                df = _load_parquet_file(local_path)
                print(f"Loaded outlines from: {local_path}")
                return df
        except Exception as e:
            errors.append(f"Local loading failed: {e}")

    # Try HF if provided
    if hf_repo_id or not local_dir:
        if hf_repo_id is None:
            hf_repo_id = HF_OUTLINES_REPO
        # For HF, typically stored in versioned subdirectory, 
        # version is always v0.0 though
        hf_path = f"{version}/data/{filename}"
        try:
            df = _download_hf_parquet(hf_repo_id, hf_path)
            print(f"Downloaded outlines from HF: {hf_repo_id}/{hf_path}")
            return df
        except Exception as e:
            errors.append(f"HuggingFace loading failed: {e}")

    raise RuntimeError(f"Failed to load outlines {filename}:\n" + "\n".join(errors))
