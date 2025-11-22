"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/19-15:26
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import pytorch_lightning as pl


# 基础抽象类（保持接口一致）
class EncoderBaseClass(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass


class DecoderBaseClass(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass


# 简单的MLP编码器
class SimpleMLPEncoder(EncoderBaseClass):
    def __init__(self, input_size: int = 784, hidden_sizes: list = [512, 256], latent_dim: int = 128):
        super().__init__()
        print(f"🔧 Encoder: input_size={input_size}, hidden_sizes={hidden_sizes}, latent_dim={latent_dim}")

        layers = []
        prev_size = input_size

        # 构建编码层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size

        # 最后一层到潜在空间
        layers.append(nn.Linear(hidden_sizes[-1], latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        return self.encoder(x)


# 简单的MLP解码器
class SimpleMLPDecoder(DecoderBaseClass):
    def __init__(self, output_size: int = 784, hidden_sizes: list = [256, 512], latent_dim: int = 128):
        super().__init__()
        print(f"🔧 Decoder: output_size={output_size}, hidden_sizes={hidden_sizes}, latent_dim={latent_dim}")

        layers = []
        prev_size = latent_dim

        # 构建解码层（与编码器对称）
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size

        # 最后一层到输出
        layers.extend([
            nn.Linear(hidden_sizes[-1], output_size),
            nn.Sigmoid()  # 输出在0-1之间
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class SimpleAutoEncoder(pl.LightningModule):
    """
    简单的MLP自编码器，确保不会出现形状错误
    """

    def __init__(
            self,
            input_size: int = 784,
            encoder_hidden_sizes: list = [512, 256],
            decoder_hidden_sizes: list = [256, 512],
            latent_dim: int = 128,
            learning_rate: float = 1e-3,
            batch_size: int = 32
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 创建编码器和解码器
        self.encoder = SimpleMLPEncoder(
            input_size=input_size,
            hidden_sizes=encoder_hidden_sizes,
            latent_dim=latent_dim
        )

        self.decoder = SimpleMLPDecoder(
            output_size=input_size,
            hidden_sizes=decoder_hidden_sizes,
            latent_dim=latent_dim
        )

        # 损失函数
        self.reconstruction_loss = nn.MSELoss()

        # 验证模型形状兼容性
        self._validate_model()

    def _validate_model(self):
        with torch.no_grad():
            # 创建测试输入
            test_input = torch.randn(2, 1, 28, 28)  # batch_size=2, channels=1, 28x28
            test_input_flat = test_input.view(2, -1)
            # 测试前向传播
            encoded = self.encoder(test_input)

            decoded = self.decoder(encoded)


    def forward(self, x):
        """前向传播"""
        # 展平输入
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        x_recon = self.decoder(z)

        # 重塑回原始形状
        return x_recon.view(x.size(0), 1, 28, 28)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)

        # 计算重建损失
        loss = self.reconstruction_loss(x_recon, x)

        self.log('train_loss', loss, prog_bar=True)
        self.log('learning_rate', self.learning_rate, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = self.reconstruction_loss(x_recon, x)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = self.reconstruction_loss(x_recon, x)

        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer