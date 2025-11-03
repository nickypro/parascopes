# %%
from huggingface_hub import HfApi, upload_folder
import os

# Initialize the API
api = HfApi()
repo = "nickypro/fineweb-llama3b-residuals"
tensor_dir = "./tensors/llama-3b"

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

print(f"Found {len(files)} tensor files to upload")

for i, file in tqdm(enumerate(files)):
    # file = "res_data_002.pt"
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