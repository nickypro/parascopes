# config.py

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

# ---------------------------------------------------------------------
# High-level dataset / generation config (unchanged from your version)
# ---------------------------------------------------------------------

HF_DATASET = "annnettte/fineweb-gemma4b-texts"
HF_SPLIT = "train"
HF_PRIVATE = True

N_SAMPLES = 5000

RESULTS_DIR = Path(__file__).parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VERSION = "5.0"
GENERATIONS_CSV = os.path.join(RESULTS_DIR, f"outlines_{VERSION}_n{N_SAMPLES}.csv")

# model to generate outlines
MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct"
]

OUTLINE_PROMPT_RULES = """
Return a short, high-level bullet-point outline of the main ideas from the text you are given.
Do NOT include any reasoning.

Rules:
- Make as 4-5 bullet points maximum
- Use numbers to enumerate the bullet points
- Aim to capture main ideas of the whole text in the bullet points
- At most 2 short subpoints per point
- Short phrases only (no lengthy sentences)
- Specific to this text (not generic).
"""

# API parameters for outline generation
OUTLINE_TEMPERATURE = 0.2
OUTLINE_MAX_TOKENS = 700

SONAR_BATCH_SIZE = 32      # Batch size for SONAR embedding generation (GPU memory dependent)
PROCESS_BATCH_SIZE = 100   # Batch size for processing samples (CPU memory dependent)

# Probe / residuals / embeddings configuration
# Small registry of models and their residual / embedding locations.
#
#   export PARASCOPES_MODEL=gemma4b
#   export PARASCOPES_MODEL=gemma27b
#   export PARASCOPES_MODEL=gemma12b
#   export PARASCOPES_MODEL=llama3b
#
# Or SELECTED_MODEL below.

MODEL_CONFIGS = {
    # Gemma 4B: local residuals + local embeddings
    "gemma4b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-4b",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/gemma4b-outlines-embeddings",
        "hf_embeds_repo": None,
    },

    # Gemma 27B: local residuals + local embeddings
    "gemma27b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-27b",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/gemma27b-outlines-embeddings",
        "hf_embeds_repo": None,
    },

    # Gemma 12B: local residuals + HF embeddings only
    "gemma12b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/gemma-12b",
        "hf_residuals_repo": None,
        "local_embeds_dir": None,
        "hf_embeds_repo": "yulia-volkova/gemma12b-outlines-embeddings",
    },

    # Llama 3B: local residuals + local embeddings
    "llama3b": {
        "local_residuals_dir": "/mnt/hdd-8tb/hdd_cache/tensors/llama-3b-new",
        "hf_residuals_repo": None,
        "local_embeds_dir": "/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results/llama-3b-outlines-embeddings_new",
        "hf_embeds_repo": None,
    },
}

# Choose which model's residuals/embeddings to use.
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

HF_OUTLINES_REPO = "yulia-volkova/parascopes-outlines-gemma4b"

print(
    f"[config] Using model '{SELECTED_MODEL}'\n"
    f"  LOCAL_RESIDUALS_DIR = {LOCAL_RESIDUALS_DIR}\n"
    f"  HF_RESIDUALS_REPO   = {HF_RESIDUALS_REPO}\n"
    f"  LOCAL_EMBEDS_DIR    = {LOCAL_EMBEDS_DIR}\n"
    f"  HF_EMBEDDINGS_REPO  = {HF_EMBEDDINGS_REPO}"
)
