# bimjepa/datasets/bimgeom_datamodule_label_efficiency.py
import os
import random
from typing import Optional, List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

def _list_npy_files(dir_path: str) -> List[str]:
    """Helper to list all .npy files in a directory."""
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")]

class BIMGEOMDataset(Dataset):
    """A PyTorch Dataset for loading .npy point clouds from the BIMGEOM structure."""
    def __init__(self, file_paths: List[str], class_to_idx: dict):
        super().__init__()
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        point_cloud = np.load(file_path).astype(np.float32)
        # Assumes folder structure: .../data_dir/ClassName/train/file.npy
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.class_to_idx.get(class_name, -1)
        return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)

class BIMGEOMDataModuleLE(pl.LightningDataModule):
    """
    BIMGEOM DataModule with training subset selection for label/data efficiency experiments.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.0,
        seed: int = 42,
        # Label Efficiency parameters
        train_fraction: float = 1.0,
        subset_seed: Optional[int] = None,
        stratified_per_class: bool = False,
        min_samples_per_class: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Placeholders
        self.class_to_idx: Dict[str, int] = {}
        self.train_dataset: Optional[BIMGEOMDataset] = None
        self.val_dataset: Optional[BIMGEOMDataset] = None
        self.test_dataset: Optional[BIMGEOMDataset] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def setup(self, stage: Optional[str] = None):
        data_dir = self.hparams.data_dir
        subset_seed = self.hparams.subset_seed if self.hparams.subset_seed is not None else self.hparams.seed
        rng = random.Random(subset_seed)

        # Discover classes from directories
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()
        self.class_to_idx = {name: i for i, name in enumerate(class_dirs)}

        # Collect files per class for training, and all test files
        train_files_per_class: Dict[str, List[str]] = {}
        test_files: List[str] = []

        for cname in class_dirs:
            class_path = os.path.join(data_dir, cname)
            
            train_path = os.path.join(class_path, 'train')
            if os.path.exists(train_path):
                files = _list_npy_files(train_path)
                files.sort()
                train_files_per_class[cname] = files

            test_path = os.path.join(class_path, 'test')
            if os.path.exists(test_path):
                files = _list_npy_files(test_path)
                files.sort()
                test_files.extend(files)

        # --- Subsetting logic ported from IFCNetCoreDataModuleLE ---
        train_fraction = float(self.hparams.train_fraction)
        assert 0.0 < train_fraction <= 1.0, "train_fraction must be in (0, 1]."

        if self.hparams.stratified_per_class:
            selected_train_files: List[str] = []
            for cname, files in train_files_per_class.items():
                files = files[:]  # copy
                rng.shuffle(files)
                if train_fraction >= 1.0:
                    k = len(files)
                else:
                    k = max(self.hparams.min_samples_per_class if len(files) > 0 else 0,
                            int(round(len(files) * train_fraction)))
                    k = min(k, len(files))
                selected_train_files.extend(files[:k])
        else:
            # Global sampling without class stratification
            all_train = [f for files in train_files_per_class.values() for f in files]
            rng.shuffle(all_train)
            k = int(round(len(all_train) * train_fraction))
            k = max(1, k) # Ensure at least one sample
            selected_train_files = all_train[:k]

        # Optional: train/val split *after* subsetting
        if self.hparams.val_split_ratio > 0.0 and len(selected_train_files) > 1:
            def label_of(fp: str) -> int:
                cname = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                return self.class_to_idx.get(cname, -1)

            y = [label_of(fp) for fp in selected_train_files]
            train_files, val_files = train_test_split(
                selected_train_files,
                test_size=self.hparams.val_split_ratio,
                random_state=self.hparams.seed,
                stratify=y if len(set(y)) > 1 else None,
            )
        else:
            train_files = selected_train_files
            val_files = []

        self.train_dataset = BIMGEOMDataset(train_files, self.class_to_idx)
        self.val_dataset = BIMGEOMDataset(val_files, self.class_to_idx)
        self.test_dataset = BIMGEOMDataset(test_files, self.class_to_idx)
        
        print("BIMGEOMDataModuleLE setup complete:")
        print(f" - Found {self.num_classes} classes.")
        print(f" - Train fraction: {train_fraction:.4f} (stratified={self.hparams.stratified_per_class})")
        print(f" - Train samples: {len(self.train_dataset)}")
        print(f" - Validation samples: {len(self.val_dataset)}")
        print(f" - Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=(self.hparams.num_workers > 0),
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers if self.val_dataset and len(self.val_dataset) > 0 else 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=(self.hparams.num_workers > 0),
        )