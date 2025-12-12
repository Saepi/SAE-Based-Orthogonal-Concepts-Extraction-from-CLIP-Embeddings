import os
import os.path as osp
import argparse
import logging

def get_args():
    parser = argparse.ArgumentParser("SAE Latent Sparse Linear Classifier")

    parser.add_argument("--train_emb", type=str, default="data/sae_embeddings/cifar10/noort/train_latent.pt",
                        help = "Path to train embeddings (.pt) from CLIP")
    parser.add_argument("--val_emb", type=str, default="data/sae_embeddings/cifar10/noort/val_latent.pt",
                        help = "Path to validation embeddings (.pt) from CLIP")
    parser.add_argument("--test_emb", type=str, default="data/sae_embeddings/cifar10/noort/test_latent.pt",
                        help = "Path to test embeddings (.pt) from CLIP")

    parser.add_argument("--model_dir", type=str, default="./model/cbm")
    parser.add_argument("--logs_dir", type=str, default="./training_logs/cbm")
    parser.add_argument("--model_name", type=str, default="cbm_model")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--l1_lambda", type=float, default=1e-4)

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    return args

def setup_logger(output_dir):
    log_dir = osp.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = osp.join(log_dir, "extract.log")

    logger = logging.getLogger("extract")
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