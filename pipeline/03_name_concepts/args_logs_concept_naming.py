import os
import os.path as osp
import argparse
import logging

def setup_logger(output_dir):
    log_dir = osp.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = osp.join(log_dir, "concept_naming.log")

    logger = logging.getLogger("concept_naming")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_format)

    f_handler = logging.FileHandler(log_file, mode="a")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(c_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def get_args():
    parser = argparse.ArgumentParser(description="Assign concept names to SAE neurons using vocab embeddings.")

    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to sparse_autoencoder_final.pt")
    parser.add_argument("--vocab_txt", type=str, required=True, help="Path to vocabulary text file (clipdissect_20k.txt)")
    parser.add_argument("--vocab_emb", type=str, required=True, help="Path to vocabulary embedding tensor (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save concept_names.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="With what name to save concept_names.csv")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")

    return parser.parse_args()
