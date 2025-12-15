import os
import numpy as np

import torch
import torch.nn.functional as F

from args_logs_concept_naming import get_args, setup_logger
import logging

import argparse
import torch.serialization
torch.serialization.add_safe_globals([argparse.Namespace])


def load_decoder_embeddings(sae_checkpoint_path: str, device: str):
    logging.info(f"Loading SAE checkpoint: {sae_checkpoint_path}")
    state = torch.load(sae_checkpoint_path, map_location=device)

    if "state_dict" in state:
        state = state["state_dict"]

    if "decoder.weight" in state:
        dec = state["decoder.weight"]
    elif "decoder.weight_orig" in state:
        dec = state["decoder.weight_orig"]
    else:
        raise KeyError(f"Decoder weight not found in checkpoint keys: {list(state.keys())}")

    logging.info(f"Decoder weight shape = {tuple(dec.shape)}")
    return dec.float()


def load_vocabulary(vocab_txt, vocab_emb, device):
    logging.info(f"Loading vocabulary names: {vocab_txt}")
    names = np.genfromtxt(vocab_txt, dtype=str, delimiter="\n")

    logging.info(f"Loading vocabulary embeddings: {vocab_emb}")
    emb = torch.load(vocab_emb, map_location=device).float()

    logging.info(f"Loaded {emb.shape[0]} vocab words with dim={emb.shape[1]}")
    return names, emb


def assign_neuron_names(decoder_emb, vocab_emb, vocab_names):
    logging.info("Normalizing and computing cosine similarities…")

    # decoder_norm = F.normalize(decoder_emb, dim=1)
    # vocab_norm = F.normalize(vocab_emb, dim=1)

    sim = decoder_emb.T @ vocab_emb.T

    logging.info("Extracting top-1 concept names for each neuron…")
    top_idx = sim.argmax(dim=1).cpu().numpy()

    assigned_names = [vocab_names[i] for i in top_idx]
    return assigned_names

def save_csv(path, names):
    logging.info(f"Saving concept names: {path}")
    with open(path, "w") as f:
        for i, name in enumerate(names):
            f.write(f"{i},{name}\n")
    logging.info("CSV saved successfully.")


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    logger.info("===== CONCEPT NAMING STARTED =====")

    decoder_emb = load_decoder_embeddings(args.sae_checkpoint, device=args.device)
    vocab_names, vocab_emb = load_vocabulary(args.vocab_txt, args.vocab_emb, device=args.device)

    neuron_names = assign_neuron_names(decoder_emb, vocab_emb, vocab_names)

    csv_path = os.path.join(args.output_dir, args.output_csv)
    save_csv(csv_path, neuron_names)

    logger.info(f"Done! Saved concept names to: {csv_path}")
    logger.info("===== CONCEPT NAMING FINISHED =====")
