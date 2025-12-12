import os
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import clip

from args_cifar10 import get_args, setup_logger


def load_clip_model(name, device):
    if name.startswith("clip-"):
        name = name[5:]
    else:
        name = name
    model, preprocess = clip.load(name, device=device)
    model.eval()
    return model, preprocess


def load_cifar10_splits(root, preprocess, batch_size, logger):
    logger.info("Loading CIFAR-10 datasets…")

    full_train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=preprocess
    )

    val_size = 5000
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, len(full_train)))

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=preprocess
    )

    logger.info(f"Train size: {len(train_set)}")
    logger.info(f"Val size:   {len(val_set)}")
    logger.info(f"Test size:  {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False,)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader


def extract_embeddings(model, loader, device, logger, split_name):
    logger.info(f"Extracting {split_name} embeddings…")

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"{split_name}"):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs).cpu()
            all_feats.append(feats)
            all_labels.append(labels)

    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    logger.info(f"{split_name} extraction completed: {feats.shape[0]} embeddings")
    return feats, labels


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    logger.info("===== CIFAR-10 CLIP Extraction Started =====")
    logger.info(f"Arguments: {vars(args)}")

    logger.info(f"Loading CLIP model: {args.clip_model}")
    model, preprocess = load_clip_model(args.clip_model, args.device)

    train_loader, val_loader, test_loader = load_cifar10_splits(
        args.data_root, preprocess, args.batch_size, logger
    )

    train_emb, train_labels = extract_embeddings(model, train_loader, args.device, logger, "train")
    torch.save(
        {"embeddings": train_emb, "labels": train_labels},
        os.path.join(args.output_dir, "train.pt"),
    )
    logger.info("Saved train embeddings.")

    val_emb, val_labels = extract_embeddings(model, val_loader, args.device, logger, "val")
    torch.save(
        {"embeddings": val_emb, "labels": val_labels},
        os.path.join(args.output_dir, "val.pt"),
    )
    logger.info("Saved val embeddings.")

    test_emb, test_labels = extract_embeddings(model, test_loader, args.device, logger, "test")
    torch.save(
        {"embeddings": test_emb, "labels": test_labels},
        os.path.join(args.output_dir, "test.pt"),
    )
    logger.info("Saved test embeddings.")

    logger.info("===== DONE. All splits extracted successfully. =====")
