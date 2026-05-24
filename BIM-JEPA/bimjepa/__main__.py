from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from pytorch_lightning.cli import LightningCLI
from bimjepa.callbacks.log_at_best_val import TrackLinearAccAtMinLossCallback
from bimjepa.models import BimJepa


from bimjepa.callbacks.wandb_checkpoint_logger import WandbModelCheckpointLogger

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    cli = LightningCLI(
        BimJepa,
        seed_everything_default=1,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )