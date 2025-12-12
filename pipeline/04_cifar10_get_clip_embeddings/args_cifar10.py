import os
import os.path as osp
import argparse
import logging

def get_args():
    parser = argparse.ArgumentParser("CIFAR-10 CLIP embedding extractor")

    parser.add_argument("--data_root", type=str, required=True,
        help="Folder where CIFAR-10 will be downloaded (or already exists)")
    
    parser.add_argument("--output_dir", type=str, required=True,
        help="Folder to save extracted embeddings")

    parser.add_argument("--clip_model", type=str, default="clip-RN50",
        help="CLIP model name, e.g., clip-RN50 or clip-ViT-B/32")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to run CLIP on")
    
    parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size for dataloader")

    return parser.parse_args()

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