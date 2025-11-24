# %%
from huggingface_hub import HfApi, upload_folder
import os

# Initialize the API
api = HfApi()
repo = "nickypro/fineweb-gemma4b-residuals"
tensor_dir = "./tensors/gemma-4b"

# Create a repository (do this once)
try:
    api.create_repo(
        repo_id=repo, 
        repo_type="dataset",
        private=False,  # or False if you want it public
    )
except Exception as e:
    print(e)

from huggingface_hub import HfApi
import os
from tqdm import tqdm

api = HfApi()
files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
files = list(sorted(files))
SKIP_EXISTING = True
print(f"Found {len(files)} tensor files to upload")

for i, file in tqdm(enumerate(files)):
    # file = "res_data_002.pt"
    if SKIP_EXISTING:
        try:
            # Check if file already exists in the repo
            api.file_exists(
                path_in_repo=file,
                repo_id=repo,
                repo_type="dataset"
            )
            print(f"⊘ Skipping {file}: already exists")
            continue
        except Exception:
            # File doesn't exist, proceed with upload
            pass

    try:
        api.upload_file(
            path_or_fileobj=os.path.join(tensor_dir, file),
            path_in_repo=file,
            repo_id=repo,  # Replace with your actual repo
            repo_type="dataset"
        )
    except Exception as e:
        print(f"✗ Failed {file}: {e}")
    break
