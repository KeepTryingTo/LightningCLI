"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-21:13
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

# mnist_cli.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class MNISTModel(pl.LightningModule):
    """
    一个简单的全连接神经网络用于MNIST分类。
    """
    def __init__(self,
                 input_size: int = 28 * 28,
                 hidden_size: int = 128,
                 num_classes: int = 10, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()  # 自动保存所有参数，便于后续加载模型
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图片
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # 使用'on_epoch_end'确保在验证周期结束时求平均
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

