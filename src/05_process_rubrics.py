import json
import os
from collections import defaultdict

INPUT_DIR = "./hdd_cache/rubric_scores/llama-3b"
OUTPUT_DIR = "./hdd_cache/processed_rubrics/llama-3b"

compare_files = {
    "TAE cat": f"{INPUT_DIR}/TAE-cat.json",
    "TAE sum": f"{INPUT_DIR}/TAE-sum.json",
    "TAE no diff": f"{INPUT_DIR}/TAE-no-diff.json",
    "TAE attn": f"{INPUT_DIR}/TAE-attn.json",
    "TAE mlp": f"{INPUT_DIR}/TAE-mlp.json",
    "auto-decoded": f"{INPUT_DIR}/auto-decoded.json",
}

metrics = [
    "complexity",
    "coherence",
    "structure",
    "subject",
    "entities",
    "details",
    "terminology",
    "tone",
]
deleted_metrics = ["identical"]

all_data = {}

for k, v in compare_files.items():
    with open(v, "r") as f:
        data = json.load(f)
        print("# loading", k)


    for index, datum_str in enumerate(data):
        if index not in all_data:
            all_data[index] = {"all_valid": True, "scores": {}}

        # load the json string into a dict
        try:
            datum = json.loads(datum_str)
        except Exception as e:
            print(index, "[json error]", [datum_str[:20] + "..." +datum_str[-20:]])
            all_data[index]["all_valid"] = False
            continue
        if 'scoring' not in datum:
            print(index, "[no scoring]", [datum_str[:20] + "..." +datum_str[-20:]])
            all_data[index]["all_valid"] = False
            continue
        for metric in metrics:
            if metric not in datum['scoring']:
                print(index, f"[no {metric} scoring]", [datum_str[:20] + "..." +datum_str[-20:]])
                all_data[index]["all_valid"] = False
                continue
        for metric in deleted_metrics:
            if metric in datum['scoring']:
                del datum['scoring'][metric]

        all_data[index]["scores"][k] = datum['scoring']


print(json.dumps(all_data[0], indent=4))

num_valid = sum(1 for v in all_data.values() if v['all_valid'])
print(f"# {len(all_data)} valid entries, of which {num_valid} are valid")

output_file = f"{OUTPUT_DIR}/all_data.json"
print(f"# writing to {output_file}")
with open(output_file, "w") as f:
    json.dump(all_data, f)