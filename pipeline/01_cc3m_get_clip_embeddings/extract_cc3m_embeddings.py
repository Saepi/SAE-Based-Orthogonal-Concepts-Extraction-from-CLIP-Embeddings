import clip
import torch

import time
import logging
import os
import os.path as osp
from tqdm import tqdm

import webdataset as wds
from PIL import Image

from args_logs import get_args, setup_logger


def load_clip(model_name: str, device: str):
    if model_name.startswith("clip-"):
        name = model_name[5:]
    else:
        name = model_name
    model, preprocess = clip.load(name, device=device)
    model.eval()
    return model, preprocess


def make_loader(shard_pattern, preprocess, batch_size, logger):
    logger.info(f"Preparing WebDataset loader for shards: {shard_pattern}")

    def count_samples(shards):
        count = 0
        for _ in wds.WebDataset(shards).decode("pil"):
            count += 1
        return count

    logger.info("Counting samples...")
    total_samples = count_samples(shard_pattern)
    logger.info(f"Total samples detected: {total_samples}")

    dataset = (
        wds.WebDataset(shard_pattern)
        .decode("pil")
        .to_tuple("jpg", "__key__")
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: (
            torch.stack([preprocess(img) for img, key in batch]),
            [key for img, key in batch],
        )
    )

    return loader, total_samples


def extract_split(name, shard_pattern, model, preprocess, device, batch_size, logger):
    logger.info(f"\n=== Processing split: {name} ===")

    loader, total_samples = make_loader(shard_pattern, preprocess, batch_size, logger)

    embeddings = []
    keys = []
    processed = 0

    with torch.no_grad():
        for imgs, batch_keys in tqdm(loader, desc=f"{name}"):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs).cpu()

            embeddings.append(feats)
            keys.extend(batch_keys)

            processed += imgs.size(0)
            if processed % (batch_size * 10) == 0:
                logger.info(f"[{name}] Processed {processed} / ~{total_samples}")

    logger.info(f"[{name}] Finished: total processed = {processed}")

    return torch.cat(embeddings, dim=0), keys

def process_split(split_name, shard_range, model, preprocess, logger, args):
    shard_pattern = osp.join(args.data_root, split_name, f"{{{shard_range}}}.tar")
    logger.info(f"Processing split: {split_name} (shards: {shard_pattern})")

    emb, keys = extract_split(
        split_name,
        shard_pattern,
        model,
        preprocess,
        args.device,
        args.batch_size,
        logger
    )

    out_path = osp.join(args.output_dir, f"{split_name}.pt")
    torch.save({"embeddings": emb, "keys": keys}, out_path)
    logger.info(f"Saved {split_name} set to {out_path}")



if __name__ == "__main__":
    args = get_args()
    logger = setup_logger(args.output_dir)
    logger.info("Starting CC3M embedding extraction.")
    logger.info(f"Arguments: {vars(args)}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading CLIP model: {args.img_enc_name} on {args.device}")
    model, preprocess = load_clip(args.img_enc_name, args.device)

    for split_name, split_name, shard_range in [
        ("train", args.train_range),
        ("validation", args.val_range),
        ("test", args.test_range)
    ]:
        process_split(split_name, shard_range, model, preprocess, logger, args)

    logger.info("DONE.")
