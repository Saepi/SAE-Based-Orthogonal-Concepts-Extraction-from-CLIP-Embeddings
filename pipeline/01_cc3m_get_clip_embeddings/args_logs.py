import os
import os.path as osp
import argparse
import logging

def get_args():
    parser = argparse.ArgumentParser("CC3M CLIP embedding extractor")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing training/*.tar and validation/*.tar")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to save embeddings")

    parser.add_argument("--train_dir", type=str, default="training")
    parser.add_argument("--val_dir", type=str, default="validation")
    parser.add_argument("--test_dir", type=str, default="test")

    parser.add_argument("--train_range", type=str, default="00000..00047",
                        help="Shard range for training set")
    parser.add_argument("--val_range", type=str, default="00000..00001",
                        help="Shard range for validation set")
    parser.add_argument("--test_range", type=str, default="00000..00001",
                        help="Shard range for test set")
    
    parser.add_argument("--img_enc_name", type=str, default="clip-RN50",
                        help="CLIP model name, e.g., clip-RN50 or clip-ViT-B/32")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)

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
