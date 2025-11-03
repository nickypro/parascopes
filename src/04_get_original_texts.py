import os
import json  # for saving output as JSON
import torch
from utils_load_data import load_res_data, load_embeds, BASE_DIR

def main():
    model_path = "llama-3b"
    os.makedirs(f"{BASE_DIR}/comparison_texts/{model_path}", exist_ok=True)

    res, paragraphs, shapes = load_res_data(999, model_path=model_path)
    # embeds = load_embeds(999, shapes, model_path=model_path).to(device, torch.float16)
    output_path = f"{BASE_DIR}/comparison_texts/{model_path}/original_texts.json"

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    flat_paragraphs = []
    for idx, (shape, paragraph_block) in enumerate(zip(shapes, paragraphs)):
        ps = paragraph_block[1:1+shape]
        flat_paragraphs.extend(ps)
        assert len(ps) == shape, f"{len(ps)=} != {shape=} for index {idx=}"

    # Save the original texts list to JSON.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flat_paragraphs, f, indent=4)

    print(f"Number of original texts: {len(flat_paragraphs)}")
    print(f"Original texts saved to: {output_path}")

if __name__ == "__main__":
    main()
