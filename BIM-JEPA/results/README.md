# Results

Training artifacts and final evaluation outputs for the three runs reported in the paper. Each subfolder corresponds to one model:

| Folder | Run | Final result |
|---|---|---|
| `pretrain/` | Self-supervised pre-training on IFC-884K + IFCNet + BIMGEOM (no-test splits) | (see `metrics.csv`) |
| `classification_ifcnetcore/` | Fine-tuning on IFCNetCore | 89.37% OA / 86.63% mean class acc |
| `classification_bimgeom/` | Fine-tuning on BIMGEOM | 92.43% OA / 89.53% mean class acc |

## Contents of each subfolder

| File | What it is |
|---|---|
| `hparams.yaml` | Resolved hyperparameters as captured by PyTorch Lightning at training start. Useful as a record of the exact configuration that produced the result. |
| `metrics.csv` | Per-step (pre-training) or final-test (fine-tuning) metrics logged by Lightning's `CSVLogger`. |
| `log.txt` | Captured stdout from the training job (PBS / local run). Includes the augmentation pipeline summary, per-epoch progress, EMA stats, and (for fine-tuning) the confusion matrix and final test metric block. |

## Reproducing these results

The configs that produced these artifacts are:

- `pretrain/` → [`configs/BIM-JEPA/pretraining/combined_pretrain_original.yaml`](../configs/BIM-JEPA/pretraining/combined_pretrain_original.yaml)
- `classification_ifcnetcore/` → [`configs/BIM-JEPA/classification/ifcnet_classification.yaml`](../configs/BIM-JEPA/classification/ifcnet_classification.yaml)
- `classification_bimgeom/` → [`configs/BIM-JEPA/classification/bimgeom_classification.yaml`](../configs/BIM-JEPA/classification/bimgeom_classification.yaml)

See the main repo [`README.md`](../../README.md) for environment setup and launch commands.

## Notes

- Paths in `log.txt` referencing the original HPC user account have been sanitized to `xxxxxxxx`.
- `metrics.csv` in `pretrain/` is the full per-step training trajectory across 500 epochs. The two fine-tuning `metrics.csv` files are single-row final test metrics.
