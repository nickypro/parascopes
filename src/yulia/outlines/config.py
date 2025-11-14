import os
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  


# HuggingFace original source data config
HF_DATASET = "annnettte/fineweb-gemma4b-texts"
HF_SPLIT = "train"
HF_PRIVATE = True

N_SAMPLES = 5000

RESULTS_DIR = Path(__file__).parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_OUTLINES_REPO = "yulia-volkova/parascopes-outlines-gemma4b"

HF_RESIDUALS_REPO = "yulia-volkova/parascopes-outlines-gemma4b"
HF_EMBEDDINGS_REPO = "yulia-volkova/gemma4b-outlines-embeddings"

# Optional local directories for chunks (can leave as None)
LOCAL_RESIDUALS_DIR = None  # e.g. Path("/data/residual_chunks")
LOCAL_EMBEDS_DIR    = None  # e.g. Path("/data/embed_chunks")

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
