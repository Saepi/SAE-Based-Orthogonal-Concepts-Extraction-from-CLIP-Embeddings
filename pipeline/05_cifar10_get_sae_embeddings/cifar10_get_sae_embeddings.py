import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pytorch_lightning as pl

from args_sae_embeddings import get_args, setup_logger

sys.path.append(os.path.abspath("pipeline/02_train_sae"))
from train_sae import SAE

import argparse
import torch.serialization
from lightning_fabric.utilities.data import AttributeDict

torch.serialization.add_safe_globals([AttributeDict])
torch.serialization.add_safe_globals([argparse.Namespace])


def load_sae_model(checkpoint_path, device):
    logger.info(f"Loading SAE model from {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("state_dict", state)

    decoder_weight = state_dict.get("decoder.weight", None)
    if decoder_weight is None:
        decoder_weight = state_dict.get("decoder.weight_orig", None)
    if decoder_weight is None:
        raise KeyError("Decoder weight not found in checkpoint")

    latent_dim = decoder_weight.shape[1]
    input_dim = decoder_weight.shape[0]

    class Args:
        def __init__(self):
            self.latent_dim = latent_dim
            self.lambda1 = 0.0
            self.lr = 0.001
            self.weight_decay = 0.0
            self.lambda_frobenius = 0.0
            self.lambda_L2 = 0.0
            self.lambda_ortsae = 0.0
            self.lambda_srip = 0.0

    args = Args()
    model = SAE(input_dim=input_dim, args=args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded SAE: input_dim={input_dim}, latent_dim={latent_dim}")
    return model


def compute_latents(embeddings, model, device, batch_size):
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []

    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc="Computing SAE latents"):
            x_batch = x_batch.to(device)
            _, z = model(x_batch)
            all_latents.append(z.cpu())

    return torch.cat(all_latents, dim=0)


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    logger.info("===== SAE Latent Extraction Started =====")
    logger.info(f"Arguments: {vars(args)}")

    model = load_sae_model(args.sae_checkpoint, args.device)

    for split_name, emb_path in [("train", args.train_emb),
                                 ("val", args.val_emb),
                                 ("test", args.test_emb)]:
        logger.info(f"Loading {split_name} embeddings from {emb_path}")
        data = torch.load(emb_path)
        embeddings = data["embeddings"]

        logger.info(f"Computing latent vectors for {split_name}")
        latents = compute_latents(embeddings, model, args.device, args.batch_size)

        out_path = os.path.join(args.output_dir, f"{split_name}_latent.pt")
        torch.save({"latents": latents, "labels": data["labels"]}, out_path)
        logger.info(f"Saved {split_name} latent vectors to {out_path}")

    logger.info("===== SAE Latent Extraction Finished =====")
