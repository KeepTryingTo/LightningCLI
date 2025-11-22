"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-21:14
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import pytorch_lightning as pl
from datas.dataset import MNISTDataModule
from models.lightningCLI import MNISTModel
from models.multi_module_lightningCLI import SimpleAutoEncoder
from pytorch_lightning.cli import LightningCLI

# pip install 'jsonargparse[signatures]>=4.17.0'

#TODO 第一点 保存模型
save_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',#TODO 监控的指标为平均绝对误差最小的,这一点和on_validation_epoch_end日志记录的指标是呼应的
    save_top_k=1, #TODO 这里的1，表示保存的模型中，只保存前4个最好结果模型权重文件
    mode='min',#TODO 表示保存当前误差最小的模型
    filename='{epoch}-{val_mae:.2f}',#TODO 保存模型格式,
    dirpath=r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCLIP/scripts/myLightningCLI/ckpt' #保存模型的路径
)

def cli_main_method_one():
    # 核心：初始化 myLightningCLI
    cli = LightningCLI(
        model_class=MNISTModel,
        datamodule_class=MNISTDataModule,
        seed_everything_default=42,  # 设置随机种子以保证可重复性
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_kwargs={'config_filename': 'default_config.yaml'},  # 将实验配置保存到文件
    )
    # 将回调添加到trainer
    cli.trainer.callbacks.append(save_callback)
    # 运行训练
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

def cli_main_method_two():
    # 核心：初始化 myLightningCLI，第二种方式是采用命令行的方式来进行
    # 回调函数的设置放在配置文件config.yaml中进行设置
    cli = LightningCLI(
        model_class=MNISTModel,
        datamodule_class=MNISTDataModule,
        seed_everything_default=42,  # 设置随机种子以保证可重复性
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_kwargs={'config_filename': 'config.yaml'},  # 将实验配置保存到文件
    )

def cli_main_method_three():
    from datas.mutl_module_dataset import MNISTDataModule
    cli = LightningCLI(
        SimpleAutoEncoder,
        MNISTDataModule,
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={'config_filename': 'mutli_module_config.yaml'}
    )


if __name__ == '__main__':
    # cli_main_method_one()
    # cli_main_method_two()
    cli_main_method_three()
    pass