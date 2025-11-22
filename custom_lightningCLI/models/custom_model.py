"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/20-15:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

# models/autoencoder.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Autoencoder(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 784,
            hidden_sizes: List[int] = [512, 256, 128],
            latent_dim: int = 64,
            learning_rate: float = 1e-3,
            batch_size: int = 32,  # 从数据模块链接过来的参数
            dropout: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 编码器
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        encoder_layers.append(nn.Linear(hidden_sizes[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        prev_size = latent_dim
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        decoder_layers.append(nn.Linear(hidden_sizes[0], input_size))
        decoder_layers.append(nn.Sigmoid())  # 输出在0-1之间
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)

        # 计算重建损失
        loss = F.mse_loss(x_recon, x.view(x.size(0), -1))

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_batch_size", self.batch_size, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = F.mse_loss(x_recon, x.view(x.size(0), -1))

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = F.mse_loss(x_recon, x.view(x.size(0), -1))

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }