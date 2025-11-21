import os
from argparse import ArgumentParser
import torch
from utils_load_data import load_res_data, BASE_DIR  # Assumption: This returns a torch.Tensor or similar.
from utils_train import Trainer         # Assumption: Trainer class handles model loading & normalization.

NUM_COPIES = 1
#PREFIX = "sonar-sweeps"
PREFIX  = "notebooks-sonar"
POSTFIX = "_099"
INDEX = 99


def main():
    parser = ArgumentParser(description='Load MLP/Linear model and infer embeds from res_data')
    parser.add_argument('wandb_run_name', type=str,
                        help='Name of the W&B run to load (e.g., northern-sweep-37)')
    args = parser.parse_args()


    # Load the trainer, which wraps the model (assumed to be a linear/MLP model)
    trainer = Trainer.load_from_wandb(PREFIX + "/" + args.wandb_run_name)
    trainer.model.eval()
    DEVICE = trainer.device
    model_type = trainer.c.model_type
    model_path = trainer.c.model_path

    # Load the full res_data 
    res_data, paragraphs, shapes = load_res_data(
        INDEX,
        groups_to_load=trainer.c.groups_to_load,
        group_size=trainer.c.group_size,
        model_path=model_path,
        group_operation=trainer.c.group_operation,
        do_diff_data=trainer.c.do_diff_data,
    )
    res_data = res_data

    # Normalize the res_data (assuming trainer.normalizer_res supports batched data)
    num_samples = res_data.shape[0]
    # Run inference through the model and restore normalization on the output embeds
    with torch.no_grad():
        # Process in batches of 1000 to avoid memory issues
        batch_size = 1024
        predicted_embeds_list = []
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_data = res_data[i:batch_end].to(DEVICE)
            normalized_data = trainer.normalizer_res(batch_data)
            
            predicted_batch = trainer.model(normalized_data)
            predicted_embeds_batch = trainer.normalizer_emb.restore(predicted_batch)
            predicted_embeds_list.append(predicted_embeds_batch.cpu())
        
        predicted_embeds = torch.cat(predicted_embeds_list, dim=0)

    # Setup the output file path and check whether it exists.
    output_dir = f"{BASE_DIR}/inferred_outputs/{model_path}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inferred_embeds_{args.wandb_run_name}{POSTFIX}_{model_type}.pt")
    if os.path.exists(output_path):
        print(f"Inferred embeds file already exists at: {output_path}. Skipping inference.")
        return

    # Save the inferred embeds output
    torch.save(predicted_embeds, output_path)
    print(f"Inferred embeds saved to: {output_path}")

if __name__ == "__main__":
    main()
