import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from args_sae import get_args
from sae_losses import frobenius_loss, ortsae_loss, srip_loss

torch.set_float32_matmul_precision('medium')
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

class SAE(pl.LightningModule):
    def __init__(self, input_dim, args):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_dim = args.latent_dim
        self.lambda1 = args.lambda1
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(self.latent_dim, input_dim)

        self.regularizers = []
        self.regularizers.append(("frob", frobenius_loss, args.lambda_frobenius))
        self.regularizers.append(("ortsae", ortsae_loss, args.lambda_ortsae))
        self.regularizers.append(("srip", srip_loss, args.lambda_srip))

    def forward(self, x):
        x = x.to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def explained_variance(self, x, x_hat):
        var_orig = torch.var(x, dim=0, unbiased=False)
        var_residual = torch.var(x - x_hat, dim=0, unbiased=False)
        ev = 1 - (var_residual / (var_orig + 1e-8))
        return ev.mean()

    def compute_losses(self, x):
        x_hat, z = self(x)
        mse_loss = F.mse_loss(x_hat, x)
        l1_loss = torch.sum(torch.abs(z))
        total_loss = mse_loss + self.lambda1 * l1_loss
        nonzero_count = torch.count_nonzero(z, dim=1).float().mean()

        reg_losses = {}
        for name, func, lam in self.regularizers:
            reg_val = func(self)
            total_loss = total_loss + lam * reg_val
            reg_losses[name] = reg_val.detach()
        return total_loss, mse_loss.detach(), l1_loss.detach(), reg_losses, nonzero_count.detach()

    def shared_step(self, batch, stage):
        x, _ = batch
        loss, mse, l1, reg_losses, nonzero_count = self.compute_losses(x)
        
        x_hat, _ = self(x)
        ev = self.explained_variance(x, x_hat)

        logs = {
            f"{stage}/loss": loss,
            f"{stage}/mse": mse,
            f"{stage}/l1": l1,
            f"{stage}/explained_variance": ev,
            f"{stage}/nonzero_z": nonzero_count
        }
        logs.update({f"{stage}/{k}": v for k, v in reg_losses.items()})
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return {"optimizer": optimizer}


class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size=256, device="cuda"):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.device = torch.device(device)

    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        self.train_data = torch.load(self.train_path)["embeddings"]
        self.val_data = torch.load(self.val_path)["embeddings"]
        self.test_data = torch.load(self.test_path)["embeddings"]

        dtype = torch.float32
        self.train_data = self.train_data.to(dtype=dtype, device=self.device)
        self.val_data = self.val_data.to(dtype=dtype, device=self.device)
        self.test_data = self.test_data.to(dtype=dtype, device=self.device)

        self.train_dataset = TensorDataset(self.train_data, torch.arange(len(self.train_data)))
        self.val_dataset = TensorDataset(self.val_data, torch.arange(len(self.val_data)))
        self.test_dataset = TensorDataset(self.test_data, torch.arange(len(self.test_data)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def main():
    args = get_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    sample_emb = torch.load(args.train_path)["embeddings"]
    input_dim = sample_emb.shape[1]

    sae_model = SAE(
        input_dim=input_dim,
        args=args
    )

    data_module = EmbeddingDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        device=args.device
    )

    tb_logger = TensorBoardLogger(
        save_dir=args.logs_dir,
        name=args.model_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=f"{args.model_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        accelerator=args.device if args.device == "cuda" else None,
        devices=1 if args.device == "cuda" else None,
        precision=32,
        log_every_n_steps=10
    )

    trainer.fit(sae_model, datamodule=data_module)
    trainer.test(sae_model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best validation model saved at {best_model_path}")

if __name__ == "__main__":
    main()
