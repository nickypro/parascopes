
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from typing import Optional
import torch


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def get_standard_features():
    """Get the standard feature schema for our dataset."""
    from datasets import Features, Value
    return Features({
        'example_id': Value('int64'),
        'dataset_idx': Value('int64'),
        'model': Value('string'),
        'completion': Value('string'),
        'outline_generated': Value('string'),
        'reconstructed_text': Value('string'),  # Always string, empty if no reconstruction
        'embedding_id': Value('int64')
    })

def load_from_hf(
    repo_id: str,
    filename: str,
    hf_token: Optional[str],
    force_download: bool = False,
):
    """Download a file from HF datasets repo and torch.load it."""
    fpath = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
        force_download=force_download,
    )
    return torch.load(fpath, map_location="cpu")


