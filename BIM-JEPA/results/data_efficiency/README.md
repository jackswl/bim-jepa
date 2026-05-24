# Data-efficiency and N-shot results

Raw per-seed results used to produce **Fig. 11 (data-efficiency curves)** and **Fig. 12 (N-shot curves)** in the paper. The plotting notebook reads these `_runs_` files directly and computes the per-point mean ± std on the fly.

## Files

| File | Sweep | Rows | Schema |
|---|---|---|---|
| `ifcnet_label_efficiency_runs.csv` | 7 fractions × 5 seeds | 35 | `tag, fraction, seed, ckpt_path, overall_accuracy, mean_class_accuracy` |
| `bimgeom_label_efficiency_runs.csv` | 7 fractions × 5 seeds | 35 | same as above |
| `ifcnet_n_shot_runs.csv` | 6 n_shot × 5 seeds | 30 | `tag, n_shot, seed, ckpt_path, overall_accuracy, mean_class_accuracy, precision, recall, f1_score` |
| `bimgeom_n_shot_runs.csv` | 6 n_shot × 5 seeds | 30 | `tag, n_shot, seed, ckpt_path, overall_accuracy, mean_class_accuracy` |

Sweep ranges:
- Label efficiency fractions: `0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00`
- IFCNetCore n-shot values: `5, 10, 18, 25, 30, 36`
- BIMGEOM n-shot values: `5, 10, 25, 50, 100, 140`
- Seeds: `42, 1337, 2026, 3105, 3407` (5 seeds per sweep point, consistent across all 4 sweeps)

## Reproducing these results

The runs were launched by the scripts in [`compute/dataefficiency_job/`](../../compute/dataefficiency_job/):

| Script | Config |
|---|---|
| `run_label_efficiency_ifcnet.py` | [`configs/BIM-JEPA/classification/ifcnet_label_efficiency.yaml`](../../configs/BIM-JEPA/classification/ifcnet_label_efficiency.yaml) |
| `run_label_efficiency_bimgeom.py` | [`configs/BIM-JEPA/classification/bimgeom_label_efficiency.yaml`](../../configs/BIM-JEPA/classification/bimgeom_label_efficiency.yaml) |
| `run_n_shot_ifcnet.py` | [`configs/BIM-JEPA/classification/ifcnet_n_shot.yaml`](../../configs/BIM-JEPA/classification/ifcnet_n_shot.yaml) |
| `run_n_shot_bimgeom.py` | [`configs/BIM-JEPA/classification/bimgeom_n_shot.yaml`](../../configs/BIM-JEPA/classification/bimgeom_n_shot.yaml) |

The pretrained encoder remained frozen for all runs; only the MLP classification head was trained for 100 epochs with a constant learning rate. See Section 4.4 of the paper for the full protocol.

## Notes

- `ckpt_path` values reference the per-run Lightning checkpoint; the HPC user id has been sanitized to `xxxxxxxx` while keeping the run version numbers for traceability against the original HPC artifacts.
- For the `bimgeom_label_efficiency`, `bimgeom_n_shot`, and `ifcnet_label_efficiency` sweeps, only `overall_accuracy` and `mean_class_accuracy` were logged per run; `ifcnet_n_shot` additionally has macro precision / recall / F1. The paper's curves only use overall accuracy and mean class accuracy (with std-dev shading over the 5 seeds).
