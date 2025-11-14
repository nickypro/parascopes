# probe_runner.py

import os
import gc
import torch
import wandb

from utils_train import Trainer
from normalizers import SAMPLES_PER_CHUNK

from config import (
    HF_RESIDUALS_REPO,
    HF_EMBEDDINGS_REPO,
    LOCAL_RESIDUALS_DIR,
    LOCAL_EMBEDS_DIR,
)
from probe_config import (
    PROBE_WANDB_PROJECT,
    get_probe_train_config,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)


def train_probe():
    try:
        # ---- 1) Base config = hyperparams from probe_config + paths from config.py ----
        base_config = get_probe_train_config()
        base_config.update(
            {
                "hf_repo_residuals": HF_RESIDUALS_REPO,
                "hf_repo_embeds": HF_EMBEDDINGS_REPO,
                "local_residuals_dir": LOCAL_RESIDUALS_DIR,
                "local_embeds_dir": LOCAL_EMBEDS_DIR,
                "prefer_local": LOCAL_RESIDUALS_DIR is not None
                                 or LOCAL_EMBEDS_DIR is not None,
            }
        )

        # ---- 2) wandb init + config ----
        wandb.init(project=PROBE_WANDB_PROJECT, config=base_config)
        cfg = dict(wandb.config)  # plain dict copy

        # ---- 3) Derived info: chunk range + norm chunks ----
        all_chunks = list(range(cfg["start_chunk"], cfg["end_chunk"] + 1))
        if not all_chunks:
            raise ValueError("No chunks selected: check start_chunk/end_chunk in probe_config.py.")

        norm_chunk_ids = all_chunks[: min(cfg["norm_chunks"], len(all_chunks))]

        print("Selected chunks (full range):", [f"{c:03d}" for c in all_chunks])
        print("Chunks used for normalization:", [f"{c:03d}" for c in norm_chunk_ids])

        # Push handy metadata to wandb
        wandb.config.update(
            {
                "all_chunks": all_chunks,
                "norm_chunk_ids": norm_chunk_ids,
                "total_chunks": len(all_chunks),
                "samples_per_chunk": SAMPLES_PER_CHUNK,
                "total_samples_full_range": len(all_chunks) * SAMPLES_PER_CHUNK,
            },
            allow_val_change=True,
        )

        # Also pass them into Trainer config
        cfg["all_chunks"] = all_chunks
        cfg["norm_chunk_ids"] = norm_chunk_ids

        # ---- 4) Create and run Trainer ----
        trainer = Trainer(cfg, DEVICE)
        model = trainer.train()

        # ---- 5) Save checkpoint with wandb metadata ----
        checkpoint_dir = cfg["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_name = f"{wandb.run.id}_{cfg['model_type']}.pkl"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        trainer.save_checkpoint(ckpt_path)

        print(f"Saved checkpoint to {ckpt_path}")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM error occurred: {e}")
            if wandb.run:
                wandb.run.finish(exit_code=1)
            gc.collect()
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()
        else:
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    train_probe()
