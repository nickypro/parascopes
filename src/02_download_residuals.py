# %%
from huggingface_hub import HfApi, hf_hub_download
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the API
api = HfApi()
repo = "nickypro/fineweb-gemma27b-residuals"
tensor_dir = "./tensors/gemma-27b"
# Download controls
start_suffix = 0   # inclusive, e.g., 0 for _000.pt
end_suffix = 99    # inclusive, e.g., 99 for _099.pt
max_workers = int(os.getenv("HF_DOWNLOAD_WORKERS", "4"))

# Create local directory if it doesn't exist
os.makedirs(tensor_dir, exist_ok=True)

from huggingface_hub import HfApi
import os
from tqdm import tqdm

api = HfApi()
# List files in the remote repository
files = [f for f in api.list_repo_files(repo_id=repo, repo_type="dataset") if f.endswith('.pt')]

# Keep only files with a numeric suffix within [start_suffix, end_suffix], e.g. *_000.pt ... *_099.pt
suffix_re = re.compile(r".*_(\d{3})\.pt$")
def in_range(filename: str) -> bool:
	match = suffix_re.match(filename)
	if not match:
		return False
	n = int(match.group(1))
	return start_suffix <= n <= end_suffix

files = [f for f in files if in_range(f)]

print(f"Found {len(files)} tensor files to download in range {start_suffix:03d}-{end_suffix:03d}")

def download_one(file: str) -> str:
	hf_hub_download(
		repo_id=repo,
		filename=file,
		repo_type="dataset",
		local_dir=tensor_dir
	)
	return file

errors = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
	future_to_file = {executor.submit(download_one, f): f for f in files}
	for _ in tqdm(as_completed(future_to_file), total=len(future_to_file)):
		pass
	for fut, file in ((fut, future_to_file[fut]) for fut in future_to_file):
		try:
			fut.result()
		except Exception as e:
			errors.append((file, str(e)))

for file, err in errors:
	print(f"✗ Failed {file}: {err}")
