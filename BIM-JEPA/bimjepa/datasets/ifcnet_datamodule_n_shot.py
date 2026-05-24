import os
import random
from typing import Optional, List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def _list_npy_files(dir_path: str) -> List[str]:
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")]


class IFCNetCoreDataset(Dataset):
    """Loads .npy point clouds and their labels from IFCNetCore-style folder structure."""
    def __init__(self, file_paths: List[str], class_to_idx: Dict[str, int]):
        super().__init__()
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        pc = np.load(file_path).astype(np.float32)
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.class_to_idx.get(class_name, -1)
        return torch.from_numpy(pc), torch.tensor(label, dtype=torch.long)


class IFCNetCoreDataModuleNShot(pl.LightningDataModule):
    """
    IFCNetCore DataModule for n-shot classification experiments.
    - n_shot: number of training samples to use per class.
    - If a class has fewer than n_shot samples, all its samples are used.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.0,
        seed: int = 42,
        n_shot: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.class_to_idx: Dict[str, int] = {}
        self.train_dataset: Optional[IFCNetCoreDataset] = None
        self.val_dataset: Optional[IFCNetCoreDataset] = None
        self.test_dataset: Optional[IFCNetCoreDataset] = None

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
            cdir = os.path.join(data_dir, cname)

            train_path = os.path.join(cdir, "train")
            if os.path.exists(train_path):
                files = _list_npy_files(train_path)
                files.sort()
                train_files_per_class[cname] = files

            test_path = os.path.join(cdir, "test")
            if os.path.exists(test_path):
                files = _list_npy_files(test_path)
                files.sort()
                test_files.extend(files)

        n_shot = self.hparams.n_shot
        selected_train_files: List[str] = []

        if n_shot is not None and n_shot > 0:
            for cname, files in train_files_per_class.items():
                files = files[:]
                rng.shuffle(files)
                # take n_shot per class, or all if the class has fewer
                k = min(n_shot, len(files))
                selected_train_files.extend(files[:k])
        else:
            print("WARNING: n_shot not specified, using full training set.")
            for files in train_files_per_class.values():
                selected_train_files.extend(files)

        if self.hparams.val_split_ratio > 0.0:
            def label_of(fp: str) -> int:
                cname = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                return self.class_to_idx[cname]

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

        self.train_dataset = IFCNetCoreDataset(train_files, self.class_to_idx)
        self.val_dataset = IFCNetCoreDataset(val_files, self.class_to_idx)
        self.test_dataset = IFCNetCoreDataset(test_files, self.class_to_idx)

        print("IFCNetCoreDataModuleNShot setup complete:")
        print(f" - Found {len(self.class_to_idx)} classes.")
        print(f" - N-Shot value: {n_shot}")
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
        nw = self.hparams.num_workers if len(self.val_dataset) > 0 else 0
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