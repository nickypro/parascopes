
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
        "train_frac": 0.9,   # global 90/10 train/val split (if you use it)
        "lr": 1e-4,
        "lr_decay": 0.8,
        "weight_decay": 1e-6,
        "d_sonar": 1024,
        "eval_every": 1000,  # steps between quick evals

        # --- chunk / data settings ---
        "start_chunk": 0,
        "end_chunk": 99,        # inclusive
        "norm_chunks": 10,      # first N chunks used for normalization
        "val_last_k": 5,        # how many last chunks to reserve for validation
        "limit_layers": None,   # or an int, e.g. 32

        # --- checkpoints / bookkeeping ---
        "checkpoint_dir": str(PROBE_CHECKPOINT_DIR),
        "resume_from": None,    # path to checkpoint if you want to resume
    }