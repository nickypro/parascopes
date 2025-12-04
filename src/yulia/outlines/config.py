# config.py - Outline Generation Configuration

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

# Outline Generation Configuration
HF_DATASET = "annnettte/fineweb-gemma12b-texts"
HF_OUTLINES_REPO = "yulia-volkova/parascopes-outlines-gemma12b"
HF_REPO_ID = HF_OUTLINES_REPO  # Alias for backward compatibility
HF_SPLIT = "train"
HF_PRIVATE = True

RESULTS_DIR = Path(__file__).parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VERSION = "0.0"
GENERATIONS_CSV = os.path.join(RESULTS_DIR, f"outlines_{VERSION}.csv")

# Model to generate outlines
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

OUTLINE_TEMPERATURE = 0.2
OUTLINE_MAX_TOKENS = 700

SONAR_BATCH_SIZE = 32      # Batch size for SONAR embedding generation (GPU memory dependent)
PROCESS_BATCH_SIZE = 100   # Batch size for processing samples (CPU memory dependent)

# Print outline generation config
print(
    f"\n{'='*60}\n"
    f"[Outline Generation Config]\n"
    f"  Source Dataset:  {HF_DATASET}\n"
    f"  Output HF Repo:  {HF_OUTLINES_REPO}\n"
    f"  Local CSV Path:  {GENERATIONS_CSV}\n"
    f"  Version:         {VERSION}\n"
    f"  Outline Model:   {MODELS[0]}\n"
    f"{'='*60}\n"
)














