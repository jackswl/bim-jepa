import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from bimjepa.datasets.ifcnet_datamodule import IFCNetCoreDataModule
from bimjepa.models import BimJepaClassification


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    cli = LightningCLI(
        BimJepaClassification,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
                # ModelCheckpoint(save_on_train_epoch_end=True),
                # ModelCheckpoint(
                #     filename="{epoch}-{step}-{val_acc:.4f}",
                #     monitor="val_acc",
                #     mode="max",
                # ),
            ],
        },
        seed_everything_default=42,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )
