import os
import textwrap
from termcolor import colored
import argparse
import json
import torch
import einops
from tqdm import tqdm

import utils_load_data
from neo_taker import Model

DEFAULT_TEST_INDEX = 99
DEFAULT_BATCH_SIZE = 8
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_NEW_TOKENS = 128
TEMP_FILE = "./comparison_texts/temp_outputs.jsonl"
os.makedirs(os.path.dirname(TEMP_FILE), exist_ok=True)

class ReplaceResidualOnce:
    """
    Replace the residual of a single token index with the given residual data.
    Returns callable class function that can be used as a hook function.

    Returns: 
        - hook_fn: (act, hook) -> act

    This is needed because model.generate() first runs the whole residual stream on
    the first n token, then only uses residual streams one by one.

    [batch, n_tokens, d_model] -> [batch, 1, d_model] -> [batch, 1, d_model] -> ...

    """
    def __init__(
            self,
            res_data,
            replace_token_index=-1,
            verbose=False,
        ):
        self.res_data = res_data
        self.replace_token_index = replace_token_index
        self.has_run = False
        self.verbose = verbose
    
    def __call__(self, act, hook):
        if self.has_run:
            if self.verbose:
                print(f"ReplaceResidualOnce: {hook.name} has already run. {act.shape=}")
            return act
        if self.verbose:
            print(f"ReplaceResidualOnce: {hook.name} running with shape {act.shape}."
                f"replacing token index {self.replace_token_index} with shape {self.res_data.shape}")

        self.has_run = True
        act[..., self.replace_token_index, :] = self.res_data
        return act

def main(verbose=False):
    parser = argparse.ArgumentParser(
        description="Generate transferred activations by directly replacing hook outputs with loaded diff segments."
    )
    parser.add_argument("--res_index", type=int, default=DEFAULT_TEST_INDEX,
                        help=f"The index used to load the res_data file (e.g. {DEFAULT_TEST_INDEX} loads tensors/res_data_{DEFAULT_TEST_INDEX:03d}.pt)")
    # Use group_size=2 to obtain interleaved diff segments for [attn, mlp]
    parser.add_argument("--group_operation", type=str, default="cat",
                        help="Group operation passed to load_res_data (should be 'cat' for concatenation, 'sum' for summation)")
    parser.add_argument("--do_diff_data", type=bool, default=False,
                        help="Whether to compute the difference between the residuals")
    parser.add_argument("--model_path", type=str, default="llama-3b",
                        help="Path to the model")
    parser.add_argument("--group_size", type=int, default=None,
                        help="Group size passed to load_res_data (should be 2 for interleaved [attn, mlp])")
    parser.add_argument("--groups_to_load", type=int, default=None,
                        help="The number of groups to load from res_data")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for the data loader")
    
    args = parser.parse_args()
    model_path = args.model_path
    if args.group_size is None:
        args.group_size = 2 if "llama" in model_path else 1

    # Determine the model repository
    repo = "meta-llama/Llama-3.2-3B-Instruct" if "llama" in model_path else model_path
    repo = "google/gemma-3-270m-it" if "gemma-270m" in model_path else repo
    repo = "google/gemma-3-1b-it" if "gemma-1b" in model_path else repo
    repo = "google/gemma-3-4b-it" if "gemma-4b" in model_path else repo
    repo = "google/gemma-3-12b-it" if "gemma-12b" in model_path else repo
    repo = "google/gemma-3-27b-it" if "gemma-27b" in model_path else repo

    # Load model
    torch.set_grad_enabled(False)
    m = Model.from_pretrained(repo, dtype="bf16")

    if args.groups_to_load is None:
        args.groups_to_load = m.cfg.n_layers

    res_layer_indices = list(range(args.groups_to_load))
    actual_layer_indices = list(range(
        m.cfg.n_layers-args.groups_to_load,
        m.cfg.n_layers,
    ))

    m.show_details()

    # To get device and token index info
    neutral_prompt = "\n\n"
    str_start_index = len(neutral_prompt)
    neutral_prompts = [neutral_prompt for _ in range(args.batch_size)]
    tokens = m.to_tokens(neutral_prompts, prepend_bos=True)
    print(f"using neutral prompt {[neutral_prompt]} * {args.batch_size}. {tokens.shape=}")
    new_token_index = tokens.shape[1] - 1
    device = tokens.device

    # Load the residual diff data (which are stored as diff values) and select one sample.
    raw_res, grouped_paragraphs, shapes = utils_load_data.load_res_data(
        args.res_index,
        model_path=model_path,
        group_operation=args.group_operation,
        do_diff_data=args.do_diff_data,
        group_size=args.group_size,
        groups_to_load=args.groups_to_load,
        flatten=False # we are not training a linear probe so no need to flatten
    )
    flat_paragraphs = []
    for idx, (shape, paragraph_block) in enumerate(zip(shapes, grouped_paragraphs)):
        ps = paragraph_block[1:1+shape]
        flat_paragraphs.extend(ps)
        assert len(ps) == shape, f"{len(ps)=} != {shape=} for index {idx=}"
    flat_paragraph_indices = torch.tensor(list(range(len(flat_paragraphs))))

    # paragraphs = utils_load_data.load_paragraphs()[-transferred_diffs.shape[0]:]
    # sample_res = raw_res.to(device)  # shape: [groups_to_load * (gdim)]
    print(f"Loaded residual data {raw_res.shape=}")
    assert len(flat_paragraphs) == raw_res.shape[0], \
        f"number of paragraphs {len(flat_paragraphs)} != number of residual data {raw_res.shape[0]}"

    # Determine the model's hidden size.
    d_model = m.cfg.d_model

    final_outputs = []
    res_dataset = torch.utils.data.TensorDataset(flat_paragraph_indices, raw_res)
    res_loader = torch.utils.data.DataLoader(res_dataset, batch_size=args.batch_size, shuffle=False)

    for [flat_paragraph_idx, batch_res] in tqdm(res_loader):
        batch_res = batch_res.to(device) # [batch, layers, gdim]
        paragraphs = [flat_paragraphs[i] for i in flat_paragraph_idx.cpu().numpy()]
        input_tokens = tokens[:batch_res.shape[0]] # if shape is less then reduce
        hook_list = []
        
        if verbose:
            print(f"adding hooks using {batch_res.shape=} to replace token index {new_token_index}")

        # For each affected decoder block, replace res_post with the injected residual.
        for data_layer_index, model_layer_index in zip(res_layer_indices, actual_layer_indices):
            layer_res_act = batch_res[:, data_layer_index]
            hook_point = f"blocks.{model_layer_index}.hook_resid_post"
            hook_fn = ReplaceResidualOnce(layer_res_act, new_token_index, verbose=verbose)
            hook_list.append((hook_point, hook_fn))

        # Generate new text with the model.
        with m.hooks(fwd_hooks=hook_list):
            gen_outputs = m.generate(
                input_ids=input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if verbose:
                print(f"generated output tokens: {gen_outputs.shape}")

            batch_outputs = []
            for (para_idx, para, gen) in zip(flat_paragraph_idx.cpu().numpy(), paragraphs, gen_outputs):
                orig = para.split('\n\n')[0]
                new  = m.to_string(gen)[str_start_index:].split("\n\n")[0]

                single_out = {
                    "model_path": model_path,
                    "res_index": int(args.res_index),
                    "para_index": int(para_idx),
                    "original": orig,
                    "generated": new,
                }
                batch_outputs.append(single_out)

                with open(TEMP_FILE, "a") as f:
                    f.write(json.dumps(single_out) + "\n")

                if verbose:
                    print(textwrap.fill(colored(f"### {para_idx} ###", "magenta"),
                                        width=120,
                                        initial_indent='',
                                        subsequent_indent=' ' * 10))
                    print(textwrap.fill(colored(f"ORIGINAL: {orig[:200]}", "blue"),
                                        width=120,
                                        initial_indent='',
                                        subsequent_indent=' ' * 10))
                    print(textwrap.fill(colored(f"GEN CONT: {new[:200]}", "green"),
                                        width=120,
                                        initial_indent='',
                                        subsequent_indent=' ' * 10))

        final_outputs.extend(batch_outputs)

    result_data = {
        "model": m.model_repo,
        "prompt": neutral_prompt,
        "res_index": args.res_index,
        # "replaced_layers": replaced_layers,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "outputs": final_outputs,
    }
    folder = f"./comparison_texts/{model_path}"
    os.makedirs(folder, exist_ok=True)
    out_filename = f"{folder}/transferred_activation_output.jsonl"
    with open(out_filename, "w") as f:
        f.write(json.dumps(result_data, indent=2) + "\n")
    print(f"Result written to {out_filename}")

if __name__ == "__main__":
    main()

