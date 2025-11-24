
from pathlib import Path

# W&B
PROBE_WANDB_PROJECT = "Separability"

# Where to save probe checkpoints
PROBE_CHECKPOINT_DIR = Path("checkpoints") / "probe_runs"
PROBE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_probe_train_config() -> dict:
    return {
        # --- model / training hyperparams ---
        "model_type": "linear_probe",
        "batch_size": 256,     
        "num_epochs": 3,
        "train_frac": 0.9,
        "lr": 1e-4,
        "lr_decay": 0.8,
        "weight_decay": 1e-6,
        "d_sonar": 1024,
        "eval_every": 1000,

        # --- chunk / data settings ---
        # 0..978 used for train+val, 979..983 held out for test
        "start_chunk": 0,
        "end_chunk": 978,      
        "norm_chunks": 10,     # first 10 for normalization
        "val_last_k": 20,      # last 20 of 0..978 for validation
        "limit_layers": None,  

        "checkpoint_dir": str(PROBE_CHECKPOINT_DIR),
        "resume_from": None,
    }