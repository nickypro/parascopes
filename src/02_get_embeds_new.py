# %%
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

torch.set_grad_enabled(False)

# %%
# Choose the dataset to mirror the splits used in get_residuals_new
# Options: "llama-3b", "gemma-4b", "gemma-12b", "gemma-27b"
model = "gemma-4b"

if model == "llama-3b":
    dataset = load_dataset("annnettte/fineweb-llama3b-texts-split")["train"]
elif model == "gemma-4b":
    dataset = load_dataset("annnettte/fineweb-gemma4b-texts-split")["train"]
elif model == "gemma-12b":
    dataset = load_dataset("annnettte/fineweb-gemma12b-texts-split")["train"]
elif model == "gemma-27b":
    dataset = load_dataset("annnettte/fineweb-gemma27b-texts-split")["train"]
else:
    raise ValueError(f"Model {model} not supported")

print("Dataset loaded:", len(dataset))
print(dataset[0])

# Initialize the SONAR model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE,
)

# %%
FOLDER, F_NAME = f"./sonar_embeds/{model}", "embeds"
os.makedirs(FOLDER, exist_ok=True)

# Determine which batch files already exist (robust resume by batch index)
existing_files = [name for name in os.listdir(FOLDER) if name.startswith(F_NAME + "_") and name.endswith(".pt")]
existing_batch_indices = set()
for name in existing_files:
    try:
        batch_num = int(name.replace(F_NAME + "_", "").replace(".pt", ""))
        existing_batch_indices.add(batch_num)
    except ValueError:
        pass

batch = []
last_seen_batch_index = None

for i, data in enumerate(tqdm(dataset)):
    batch_index = i // 1000

    # Skip if this batch already exists
    if batch_index in existing_batch_indices:
        continue

    # Save completed batch
    if i > 0 and i % 1000 == 0:
        save_index = batch_index - 1
        torch.save(batch, f"{FOLDER}/{F_NAME}_{save_index:03d}.pt")
        existing_batch_indices.add(save_index)
        batch = []

    # Use SONAR model to get embeddings for the output segments only
    texts = data["split_text"][1:]
    try:
        embeddings = t2vec_model.predict(texts, source_lang="eng_Latn")
    except Exception:
        print(data["split_text"])
        raise

    batch.append(embeddings)
    last_seen_batch_index = batch_index

# Save the final (possibly partial) batch
if batch and last_seen_batch_index is not None and last_seen_batch_index not in existing_batch_indices:
    torch.save(batch, f"{FOLDER}/{F_NAME}_{last_seen_batch_index:03d}.pt")

# %%