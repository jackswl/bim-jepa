# bimjepa/datasets/bimgeom_datamodule_n_shot.py
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
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.class_to_idx.get(class_name, -1)
        return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)

class BIMGEOMDataModuleNShot(pl.LightningDataModule):
    """
    BIMGEOM DataModule with training subset selection for n-shot experiments.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.0,
        seed: int = 42,
        # N-Shot parameters
        samples_per_class: int = 50,
        subset_seed: Optional[int] = None,
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

        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()
        self.class_to_idx = {name: i for i, name in enumerate(class_dirs)}

        train_files_per_class: Dict[str, List[str]] = {}
        test_files: List[str] = []

        for cname in class_dirs:
            class_path = os.path.join(data_dir, cname)
            train_path = os.path.join(class_path, 'train')
            if os.path.exists(train_path):
                train_files_per_class[cname] = sorted(_list_npy_files(train_path))
            test_path = os.path.join(class_path, 'test')
            if os.path.exists(test_path):
                test_files.extend(sorted(_list_npy_files(test_path)))

        # **MODIFIED LOGIC: N-Shot Subsetting**
        n_samples = self.hparams.samples_per_class
        assert n_samples > 0, "samples_per_class must be positive."

        selected_train_files: List[str] = []
        for cname, files in train_files_per_class.items():
            files_copy = files[:]  # Create a copy to shuffle
            rng.shuffle(files_copy)
            # Take at most n_samples, or all if the class has fewer
            k = min(n_samples, len(files_copy))
            selected_train_files.extend(files_copy[:k])
        
        # Shuffle the final selection to mix classes before creating the dataset
        rng.shuffle(selected_train_files)

        # Optional: train/val split after subsetting
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
        
        print("BIMGEOMDataModuleNShot setup complete:")
        print(f" - Found {self.num_classes} classes.")
        print(f" - Samples per class (n-shot): {n_samples}")
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