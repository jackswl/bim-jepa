import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

class TrackLinearAccAtMinLossCallback(Callback):
    def __init__(self, monitor_metric_name):
        super().__init__()
        self.best_val_loss = float('inf')
        self.monitor_metric_name = monitor_metric_name # added, as well as in the __init__

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # Safely get the current validation loss from the trainer's metrics
        val_loss_tensor = trainer.callback_metrics.get('val_loss')

        # If 'val_loss' is not available for any reason, do nothing.
        if val_loss_tensor is None:
            return

        current_val_loss = val_loss_tensor.item()

        # Check if the current validation loss is the best we've seen so far
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            pl_module.log("best_val_loss", self.best_val_loss, on_step=False, on_epoch=True)

            # Define the specific metric name for your target dataset
            # IMPORTANT: Make sure 'ifcnetcore_val' matches the key in your YAML file
            metric_name = "svm_val_acc_ifcnetcore_val"
            
            # Safely get the accuracy metric for the IFCNetCore dataset
            # svm_acc_tensor = trainer.callback_metrics.get(metric_name)
            svm_acc_tensor = trainer.callback_metrics.get(self.monitor_metric_name)
            
            # If the accuracy metric exists, log its "best" version
            # if svm_acc_tensor is not None:
            #     best_metric_name = f"best_{metric_name}"
            #     svm_accuracy = svm_acc_tensor.item()
            #     pl_module.log(best_metric_name, svm_accuracy, on_step=False, on_epoch=True)
            if svm_acc_tensor is not None:
                best_metric_name = f"best_{self.monitor_metric_name}"
                svm_accuracy = svm_acc_tensor.item()
                pl_module.log(best_metric_name, svm_accuracy, on_step=False, on_epoch=True, sync_dist=True) # sync_dist for multi-GPU

# from pytorch_lightning.callbacks import Callback
# from pytorch_lightning import Trainer, LightningModule

# class TrackLinearAccAtMinLossCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.best_val_loss = float('inf')
#         self.acc_val_acc_modelnet40_at_min_loss = 0.0
#         self.acc_val_acc_scanobjectnn_at_min_loss = 0.0

#     def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
#         # Assume 'val_loss' and 'svm_val_acc' are logged using `self.log` in your LightningModule
#         total_epochs = trainer.max_epochs
#         current_epoch = trainer.current_epoch
#         should_track_best = current_epoch >= total_epochs * 0.5

#         # Only track the best validation loss after the halfway point as the earlier epochs produce low validation loss
#         if should_track_best:
#             current_val_loss = trainer.callback_metrics.get('val_loss').item()
#             svm_val_acc_modelnet40 = trainer.callback_metrics.get('svm_val_acc_modelnet40').item()
#             svm_val_acc_scanobjectnn = trainer.callback_metrics.get('svm_val_acc_scanobjectnn').item()

#             # Halfway through the training, start tracking the best validation loss
#             if current_val_loss < self.best_val_loss: 
#                 self.best_val_loss = current_val_loss
#                 self.acc_val_acc_modelnet40_at_min_loss = svm_val_acc_modelnet40
#                 self.acc_val_acc_scanobjectnn_at_min_loss = svm_val_acc_scanobjectnn
                
#             pl_module.log("svm_val_acc_modelnet40_at_min_loss", self.acc_val_acc_modelnet40_at_min_loss)
#             pl_module.log("svm_val_acc_scanobjectnn_at_min_loss", self.acc_val_acc_scanobjectnn_at_min_loss)

#             pl_module.log("best_val_loss", self.best_val_loss)
