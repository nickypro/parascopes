# probe_runner.py

import os
import gc
from datetime import datetime
import torch
import wandb

from utils_train import Trainer
from utils_normalizers import SAMPLES_PER_CHUNK

from config_probe import (
    HF_RESIDUALS_REPO,
    HF_EMBEDDINGS_REPO,
    LOCAL_RESIDUALS_DIR,
    LOCAL_EMBEDS_DIR,
    PROBE_WANDB_PROJECT,
    get_probe_train_config,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)


def _infer_model_name_from_paths() -> str:

    env_name = os.environ.get("PARASCOPES_MODEL")
    if env_name:
        return env_name.lower()

    candidates = [
        HF_RESIDUALS_REPO,
        HF_EMBEDDINGS_REPO,
        LOCAL_RESIDUALS_DIR,
        LOCAL_EMBEDS_DIR,
    ]
    candidates = [str(c).lower() for c in candidates if c is not None]

    for candidate in candidates:
        if "gemma" in candidate:
            tail = os.path.basename(candidate)
            return tail or "gemma"
        if "llama" in candidate:
            tail = os.path.basename(candidate)
            return tail or "llama"


    return "unknown-model"


def train_probe():
    try:
        # 1) Base config = hyperparams from probe_config + paths from config.py 
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

        model_name = _infer_model_name_from_paths()
        base_config["model_name"] = model_name  # keep name in config 
        
        # Add timestamp to run name for easy tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{base_config['model_type']}_{timestamp}"

        # 2) wandb init + config 
        wandb.init(
            project="outlines_probes", 
            entity="seperability",
            config=base_config,
            name=run_name,
        )
        
        # Force update critical config values to prevent wandb from using cached old values (?)
        wandb.config.update(
            {
                "start_chunk": base_config["start_chunk"],
                "end_chunk": base_config["end_chunk"],
                "val_last_k": base_config["val_last_k"],
            },
            allow_val_change=True,
        )
        
        cfg = dict(wandb.config)  
        
        print(f"\n{'='*60}")
        print(f"CONFIG CHECK:")
        print(f"  start_chunk: {cfg['start_chunk']}")
        print(f"  end_chunk: {cfg['end_chunk']}")
        print(f"  val_last_k: {cfg['val_last_k']}")
        print(f"{'='*60}\n")

        # 3) Derived info: chunk range + norm chunk
        all_chunks = list(range(cfg["start_chunk"], cfg["end_chunk"] + 1))
        if not all_chunks:
            raise ValueError("No chunks selected: check start_chunk/end_chunk in probe_config.py.")

        norm_chunk_ids = all_chunks[: min(cfg["norm_chunks"], len(all_chunks))]

        print(f"Selected model for this run: {cfg.get('model_name', 'unknown')}")
        print("Selected chunks (full range):", [f"{c:03d}" for c in all_chunks])
        print("Chunks used for normalization:", [f"{c:03d}" for c in norm_chunk_ids])

        # Setup checkpoint directory if local saving is enabled
        save_locally = cfg.get("save_checkpoints", False)
        save_to_wandb = cfg.get("save_checkpoints_to_wandb", True)
        run_checkpoint_dir = None
        
        if save_locally:
            base_checkpoint_dir = cfg["checkpoint_dir"]
            run_checkpoint_dir = os.path.join(base_checkpoint_dir, model_name, wandb.run.id)
            os.makedirs(run_checkpoint_dir, exist_ok=True)
            cfg["checkpoint_dir"] = run_checkpoint_dir
            print(f"\nLocal checkpoints: {run_checkpoint_dir}")
        
        if save_to_wandb:
            print(f"Checkpoints will be saved to wandb as artifacts")
        
        if not save_locally and not save_to_wandb:
            print(f"No checkpoint saving enabled")
        
        # Push metadata to wandb
        wandb_metadata = {
            "all_chunks": all_chunks,
            "norm_chunk_ids": norm_chunk_ids,
            "total_chunks": len(all_chunks),
            "samples_per_chunk": SAMPLES_PER_CHUNK,
            "total_samples_full_range": len(all_chunks) * SAMPLES_PER_CHUNK,
        }
        if run_checkpoint_dir:
            wandb_metadata["checkpoint_dir"] = run_checkpoint_dir
        
        wandb.config.update(wandb_metadata, allow_val_change=True)

        # Also pass them into Trainer config
        cfg["all_chunks"] = all_chunks
        cfg["norm_chunk_ids"] = norm_chunk_ids

        #Create and run Trainer (with optional resume)
        start_epoch = 0
        resume_from = cfg.get("resume_from")
        
        if resume_from and os.path.exists(resume_from):
            print(f"\nResuming from checkpoint: {resume_from}")
            trainer, start_epoch = Trainer.load_checkpoint(resume_from, DEVICE)
            # Update config in case it changed
            trainer._config.update(cfg)
        else:
            if resume_from:
                print(f"WARNING: Checkpoint not found: {resume_from}, starting fresh")
            trainer = Trainer(cfg, DEVICE)
        
        model = trainer.train(start_epoch=start_epoch)

        # 6) Save final checkpoint ----
        save_to_wandb = cfg.get("save_checkpoints_to_wandb", True)
        
        if save_to_wandb:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
                trainer.save_checkpoint(tmp.name)
                tmp_path = tmp.name
            
            artifact = wandb.Artifact(
                name="checkpoint_final",
                type="model",
                description="Final model checkpoint after all epochs"
            )
            artifact.add_file(tmp_path)
            wandb.log_artifact(artifact)
            
            # Clean up temp file
            os.remove(tmp_path)
            print(f"\nTraining complete! Final checkpoint saved to wandb")
        else:
            # Save locally if wandb saving is disabled
            if cfg.get("save_checkpoints", False) and run_checkpoint_dir:
                final_ckpt = os.path.join(run_checkpoint_dir, "checkpoint_final.pkl")
                trainer.save_checkpoint(final_ckpt)
                print(f"\nTraining complete! Final checkpoint: {final_ckpt}")
            else:
                print(f"\nTraining complete! (No checkpoints saved)")

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
