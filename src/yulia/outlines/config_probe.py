
import os
from pathlib import Path

PROBE_WANDB_PROJECT = "Separability"

# Where to save probe checkpoints - use absolute path
_BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to parascopes/
PROBE_CHECKPOINT_DIR = _BASE_DIR / "checkpoints" / "probe_runs"
PROBE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# (Residuals + Embeddings Locations )
MODEL_CONFIGS = {
    "gemma4b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-4b",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/gemma4b-outlines-embeddings",
        "hf_embeds_repo": None,
    },

    "gemma27b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-27b",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/gemma27b-outlines-embeddings",
        "hf_embeds_repo": None,
    },

    "gemma12b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-12b",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/gemma12b-outlines-embeddings",
        "hf_embeds_repo": "yulia-volkova/gemma12b-outlines-embeddings",
    },

    "llama3b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/llama-3b-new",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/llama-3b-outlines-embeddings_new",
        "hf_embeds_repo": None,
    },
}

SELECTED_MODEL = os.environ.get("PARASCOPES_MODEL", "gemma4b").lower()
if SELECTED_MODEL not in MODEL_CONFIGS:
    raise ValueError(
        f"Unknown PARASCOPES_MODEL='{SELECTED_MODEL}'. "
        f"Valid options: {', '.join(MODEL_CONFIGS.keys())}"
    )

_selected = MODEL_CONFIGS[SELECTED_MODEL]
LOCAL_RESIDUALS_DIR = _selected["local_residuals_dir"]
HF_RESIDUALS_REPO = _selected["hf_residuals_repo"]
LOCAL_EMBEDS_DIR = _selected["local_embeds_dir"]
HF_EMBEDDINGS_REPO = _selected["hf_embeds_repo"]

# Print probe config
print(
    f"\n{'='*60}\n"
    f"[Probe Training Config]\n"
    f"  Model:              {SELECTED_MODEL}\n"
    f"  LOCAL_RESIDUALS:    {LOCAL_RESIDUALS_DIR}\n"
    f"  HF_RESIDUALS:       {HF_RESIDUALS_REPO}\n"
    f"  LOCAL_EMBEDS:       {LOCAL_EMBEDS_DIR}\n"
    f"  HF_EMBEDS:          {HF_EMBEDDINGS_REPO}\n"
    f"  Checkpoint Dir:     {PROBE_CHECKPOINT_DIR}\n"
    f"{'='*60}\n"
)


def get_probe_train_config() -> dict:
    return {
        # training hyperparams ---
        "model_type": "linear_probe",
        "batch_size": 256,     
        "num_epochs": 10,
        "lr": 2e-5,           
        "lr_decay": 0.8,
        "weight_decay": 2e-5,  
        "d_sonar": 1024,
        "log_every_n_steps": 50,  # Log every 50 steps for training curves
        # Chunks 0..98 = train, 99 = val
        "start_chunk": 0,
        "end_chunk": 99,      
        "norm_chunks": 10,      
        "val_last_k": 1,       
        "limit_layers": 62,   
        "chunks_per_epoch": None,
        "val_every_chunks": None,   
        "probe_val_chunk": None,
        # Checkpoint settings
        "save_checkpoints": False,           
        "save_checkpoints_to_wandb": True,   
        "checkpoint_dir": str(PROBE_CHECKPOINT_DIR),
        # To resume from wandb: download artifact first, then set path
        # e.g., "resume_from": "artifacts/checkpoint_epoch_1.pkl"
        "resume_from": None,
    }
