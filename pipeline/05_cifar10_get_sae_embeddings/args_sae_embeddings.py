import os
import os.path as osp
import argparse
import logging

def get_args():
    parser = argparse.ArgumentParser("SAE Latent Extraction from CLIP embeddings")

    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help = "Path to the trained SAE checkpoint (.ckpt or .pt)")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help = "Directory to save SAE latent vectors")
    
    
    parser.add_argument("--train_emb", type=str, default="data/embeddings/cifar10/train.pt",
                        help = "Path to train embeddings (.pt) from CLIP")
    
    parser.add_argument("--val_emb", type=str, default="data/embeddings/cifar10/val.pt",
                        help = "Path to validation embeddings (.pt) from CLIP")
    
    parser.add_argument("--test_emb", type=str, default="data/embeddings/cifar10/test.pt",
                        help = "Path to test embeddings (.pt) from CLIP")
    
    

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help = "Device to run SAE encoder on")
    
    parser.add_argument("--batch_size", type=int, default=256,
                        help = "Batch size for computing SAE latent vectors")


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
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
