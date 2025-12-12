import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from args_train_cbm import get_args, setup_logger

class LatentDataModule(pl.LightningDataModule):
    def __init__(self, train_emb, val_emb, test_emb, batch_size=256):
        super().__init__()
        self.train_emb = train_emb
        self.val_emb = val_emb
        self.test_emb = test_emb
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_data = torch.load(self.train_emb)
        val_data = torch.load(self.val_emb)
        test_data = torch.load(self.test_emb)

        self.train_dataset = TensorDataset(train_data["latents"], train_data["labels"])
        self.val_dataset = TensorDataset(val_data["latents"], val_data["labels"])
        self.test_dataset = TensorDataset(test_data["latents"], test_data["labels"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class SparseLinearClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes=10, lr=1e-3, weight_decay=0.0, l1_lambda=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, num_classes)

        self.val_labels_epoch = []
        self.val_preds_epoch = []
        self.test_labels_epoch = []
        self.test_preds_epoch = []

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, x, y):
        logits = self(x)
        ce_loss = F.cross_entropy(logits, y)
        l1_loss = torch.sum(torch.abs(self.linear.weight))
        total_loss = ce_loss + self.hparams.l1_lambda * l1_loss
        return total_loss, ce_loss, l1_loss, logits

    def shared_step(self, batch, stage):
        x, y = batch
        total_loss, ce_loss, l1_loss, logits = self.compute_loss(x, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}/total_loss", total_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/sparsity", l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/accuracy", acc, on_step=False, on_epoch=True)

        if stage == "val":
            self.val_labels_epoch.append(y.cpu())
            self.val_preds_epoch.append(preds.cpu())
        elif stage == "test":
            self.test_labels_epoch.append(y.cpu())
            self.test_preds_epoch.append(preds.cpu())

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def on_validation_epoch_end(self):
        labels = torch.cat(self.val_labels_epoch)
        preds = torch.cat(self.val_preds_epoch)
        cm = confusion_matrix(labels.numpy(), preds.numpy())
        fig = self.plot_confusion_matrix(cm)
        self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
        plt.close(fig)
        # reset lists
        self.val_labels_epoch = []
        self.val_preds_epoch = []

    def on_test_epoch_end(self):
        labels = torch.cat(self.test_labels_epoch)
        preds = torch.cat(self.test_preds_epoch)
        cm = confusion_matrix(labels.numpy(), preds.numpy())
        fig = self.plot_confusion_matrix(cm)
        self.logger.experiment.add_figure("test/confusion_matrix", fig, self.current_epoch)
        plt.close(fig)
        self.test_labels_epoch = []
        self.test_preds_epoch = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @staticmethod
    def plot_confusion_matrix(cm):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return fig


if __name__ == "__main__":
    args = get_args()
    logger = setup_logger(args.logs_dir)
    logger.info("===== Sparse Linear Classifier Training Started =====")

    tb_logger = TensorBoardLogger(save_dir=args.logs_dir, 
                                  name=args.model_name)

    data_module = LatentDataModule(
        train_emb=args.train_emb,
        val_emb=args.val_emb,
        test_emb=args.test_emb,
        batch_size=args.batch_size
    )

    sample_data = torch.load(args.train_emb)["latents"]
    input_dim = sample_data.shape[1]

    model = SparseLinearClassifier(
        input_dim=input_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=f"{args.model_name}-{{epoch:02d}}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1 if args.device=="cuda" else None,
        accelerator="gpu" if args.device=="cuda" else "cpu",
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    logger.info("===== Training Finished =====")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
