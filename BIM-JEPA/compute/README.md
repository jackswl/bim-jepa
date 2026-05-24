# Compute scripts

This directory contains example PBS batch scripts used to run BIM-JEPA pre-training, fine-tuning, and data-efficiency experiments on an HPC cluster. They are provided as **reference / templates** — they are not intended to run out-of-the-box on every cluster.

## Adapting the scripts for your environment

Before submitting any script with `qsub`, you will need to edit at least:

| What | Where | Notes |
|---|---|---|
| Project account | `#PBS -P personal-xxxxxxxx` | Replace `xxxxxxxx` with your HPC project / account code. |
| Conda env name | `conda activate bimjepa` | Replace if your env is named differently. |
| Module loads | `module load miniforge3` | Replace with the module name available on your cluster. |
| Resource request | `#PBS -l select=...` , `#PBS -l walltime=...` | Adjust GPU count, walltime, queue, etc. as needed. |
| Checkpoint paths | `--ckpt_path /home/users/nus/xxxxxxxx/...` | Update to your local checkpoint paths. The `xxxxxxxx` placeholder is a sanitized user id from the original environment. |
| `--artifacts_root` (data-efficiency runs) | `run_*.py` argparse default and `run_*.pbs` flag | Point to where you want training artifacts written. |

If your cluster uses Slurm instead of PBS, port the directives (`#PBS -l` → `#SBATCH`, `qsub` → `sbatch`, etc.) but keep the same `python -m bimjepa[.tasks.classification] fit -c ...` command.

## Layout

```
compute/
├── pretrain_job/          # Self-supervised pre-training on the combined dataset
├── finetune_job/          # Classification fine-tuning on IFCNetCore / BIMGEOM
└── dataefficiency_job/    # Label-efficiency curves and n-shot evaluations
```

## Running locally (single-node, no scheduler)

Most of these workloads can also be launched directly without PBS. From the `BIM-JEPA/` directory:

```
# Pre-training
python -m bimjepa fit -c configs/BIM-JEPA/pretraining/combined_pretrain_original.yaml

# Fine-tuning (after downloading the pre-trained ckpt — see main README)
python -m bimjepa.tasks.classification fit \
    -c configs/BIM-JEPA/classification/ifcnet_classification.yaml
```
