# %%
from transformer_lens import HookedTransformer
from datasets import load_dataset
import json
import torch
from tqdm import tqdm
import numpy as np
torch.set_grad_enabled(False)

MAX_TOKENS = 2048

# %%
model = "llama-3b"

if model == "llama-3b":
    dataset = load_dataset("annnettte/fineweb-llama3b-texts-split")["train"]
    m = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", dtype=torch.bfloat16)

elif model == "gemma-27b":
    from datasets import load_dataset
    dataset = load_dataset("annnettte/fineweb-gemma27b-texts-split")["train"]
    m = HookedTransformer.from_pretrained("google/gemma-3-27b-it", dtype=torch.bfloat16)

else:
    raise ValueError(f"Model {model} not supported")

print("Dataset loaded:", len(dataset))
print(dataset[0])

# %%
def format_prompt(prompt: list[str] | str, system_prompt: str = None) -> list[str]:
    def format_prompt_string(prompt: str) -> str:
        """Format prompt using the model's chat template."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return m.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    if isinstance(prompt, str):
        return [format_prompt_string(prompt)]
    elif isinstance(prompt, list):
        return [format_prompt_string(p) for p in prompt]

def get_act_data(split_text, act_types=None, verbose=False):
    # choose which residual data to collect
    if act_types is None:
        act_types = ["hook_resid_pre", "hook_resid_mid", "hook_resid_post"]
    hook_names = [
        f"blocks.{i}.{resid_type}"
            for i in range(m.cfg.n_layers)
            for resid_type in act_types
    ]

    # get prompt vs output separately
    prompt = [x+"\n" for x in format_prompt(split_text[0])]
    output = split_text[1:]

    # Tokenize the prompt and output correctly
    prompt_tokens =  m.to_tokens(prompt, prepend_bos=False)
    output_tokens = [m.to_tokens(o, prepend_bos=False) for o in output]
    if verbose:
        print(prompt_tokens.shape, [o.shape for o in output_tokens])
    all_tokens = torch.cat([prompt_tokens, torch.cat(output_tokens, dim=1)], dim=1)

    if verbose:
        print(m.to_str_tokens(all_tokens))

    # Get the indices of the residual streams that we want to store
    # Ie: last token of each section, usually "\n\n"
    final_indices_rel = [
        prompt_tokens.shape[-1],
        *[ o.shape[-1] for o in output_tokens ]
    ]
    final_indices_abs = np.cumsum(final_indices_rel) - 1

    # truncate to 8092 tokens
    if verbose:
        # Filter final indices to only include those within the truncated sequence
        print(f"{final_indices_abs=}")

    all_tokens = all_tokens[:, :MAX_TOKENS]
    final_indices_abs = final_indices_abs[final_indices_abs < MAX_TOKENS]

    if verbose:
        print(f"{final_indices_abs=}")

    # check the tokens are actually the newline ones
    if verbose:
        print(m.to_str_tokens(all_tokens[:, final_indices_abs]))

    # Create hooks to store activations of only the correct residual streams
    act_data = {}
    def store_act(act, hook):
        act_data[hook.name] = act[..., final_indices_abs, :]
    hook_list = [(name, store_act) for name in hook_names]

    # Run model and store activations
    with m.hooks(fwd_hooks=hook_list):
        m.forward(all_tokens)

    # Print some info
    if verbose:
        for k, v in act_data.items():
            print(k, v.shape)
            break

    return act_data

# %%
import os
os.makedirs(f"./tensors/{model}", exist_ok=True)

import os

# Determine how many files exist in ./tensors
existing_files = len([name for name in os.listdir(f"./tensors/{model}") if name.startswith("res_data_") and name.endswith(".pt")])
skip_count = existing_files * 1000

batch = []
for i, data in enumerate(tqdm(dataset)):
    if i < skip_count:
        continue
    if i > 0 and i % 1000 == 0:
        batch_index = i // 1000 - 1
        torch.save(batch, f"./tensors/{model}/res_data_{batch_index:03d}.pt")
        batch = []

    act_data = get_act_data(data["split_text"], verbose=False)

    # print(act_data.keys())

    res = []
    res.append(act_data["blocks.0.hook_resid_pre"])
    for i in range(m.cfg.n_layers):
        res.append(act_data[f"blocks.{i}.hook_resid_mid"])
        res.append(act_data[f"blocks.{i}.hook_resid_post"])

    # print([x.shape for x in res])
    res_tensor = torch.cat(res, dim=0) # we don't need the last residual stream

    # print(res_tensor)
    batch.append({
        "index": data["index"],
        "split_text": data["split_text"],
        "res": res_tensor.cpu(),
    })

    # print(res_tensor.shape)

batch_index += 1
torch.save(batch, f"./tensors/{model}/res_data_{batch_index:03d}.pt")

# %%
