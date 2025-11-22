"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/20-15:33
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

# train.py
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         EarlyStopping,
                                         LearningRateMonitor)
import datetime
import time

from models.custom_model import Autoencoder
from datas.custom_dataset import MNISTDataModule
from utils.send_email import (send_training_start_notification,
                              send_training_end_notification)

# 链接参数：数据模块的image_size -> 模型的input_size（需要转换）
def image_size_to_input_size(image_size):
    return image_size * image_size

class MyLightningCLI(LightningCLI):
    """
    自定义 LightningCLI 实现
    包含参数链接、训练通知等高级功能
    """

    def add_arguments_to_parser(self, parser):
        """添加自定义命令行参数"""
        # 添加通知邮箱参数
        parser.add_argument(
            "--notification_email",
            type=str,
            default="your_email@example.com",
            help="Email address for training notifications"
        )

        # 添加实验名称参数
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="autoencoder_experiment",
            help="Name of the current experiment"
        )

        # 链接参数：数据模块的batch_size -> 模型的batch_size
        parser.link_arguments(source="data.init_args.batch_size", target="model.init_args.batch_size")

        parser.link_arguments(
            source="data.init_args.image_size",
            target="model.init_args.input_size",
            compute_fn=image_size_to_input_size
        )

    def before_fit(self):
        """在训练开始前执行"""
        self.start_time = time.time()

        # 发送训练开始通知
        # send_training_start_notification(
        #     email=self.config[self.subcommand].notification_email,
        #     model_name=self.model.__class__.__name__,
        #     dataset_name="MNIST"
        # )

        # 添加回调函数
        callbacks = self._setup_callbacks()
        for callback in callbacks:
            self.trainer.callbacks.append(callback)

        print(f"Starting training for experiment: {self.config[self.subcommand].experiment_name}")
        print(f"Notifications will be sent to: {self.config[self.subcommand].notification_email}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Data: {self.datamodule.__class__.__name__}")

    def after_fit(self):
        """在训练结束后执行"""
        training_time = time.time() - self.start_time

        # 收集最终指标
        final_metrics = {}
        if hasattr(self.trainer, 'callback_metrics'):
            for key, value in self.trainer.callback_metrics.items():
                if hasattr(value, 'item'):
                    final_metrics[key] = value.item()

        # 发送训练完成通知
        # send_training_end_notification(
        #     email=self.config[self.subcommand].notification_email,
        #     model_name=self.model.__class__.__name__,
        #     training_time=str(datetime.timedelta(seconds=int(training_time))),
        #     final_metrics=final_metrics
        # )

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final metrics: {final_metrics}")

    def _setup_callbacks(self):
        """设置训练回调函数"""
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"checkpoints/{self.config[self.subcommand].experiment_name}",
            filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
            save_last=True
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        return [checkpoint_callback, early_stopping, lr_monitor]


def main():
    """主函数"""
    cli = MyLightningCLI(
        model_class=Autoencoder,
        datamodule_class=MNISTDataModule,
        seed_everything_default=42,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": True
        }
    )

    # 手动运行训练
    cli.before_fit()
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.after_fit()

    # 测试模型
    if hasattr(cli.datamodule, 'test_dataloader'):
        cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()