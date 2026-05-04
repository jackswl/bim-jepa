import os
from typing import Optional, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class IFCFullDataset(Dataset):
    """
    Dataset for loading .npy point clouds where the class name is in the filename.
    """
    def __init__(self, file_paths: List[str], class_to_idx: dict):
        super().__init__()
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        point_cloud = np.load(file_path).astype(np.float32)
        
        # Extracts class name from the filename
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        try:
            class_name = name_without_ext.split('_', 1)[1]
        except IndexError:
            class_name = "unknown"
        
        label = self.class_to_idx.get(class_name, -1)
        return torch.from_numpy(point_cloud), torch.tensor(label, dtype=torch.long)

class IFCFullDataModule(pl.LightningDataModule):
    """
    DataModule for the massive, flat-structured IFC dataset from multiple folders.
    This version strictly follows the original file's logic without stratification.
    """
    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        batch_size: int = 32,
        num_workers: int = 8,
        val_split_ratio: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # 1. Aggregate files from all specified directories.
        all_files = []
        dirs_to_scan = self.hparams.data_dirs
        if isinstance(dirs_to_scan, str):
            dirs_to_scan = [dirs_to_scan]

        for directory in dirs_to_scan:
            if os.path.isdir(directory):
                files_in_dir = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
                all_files.extend(files_in_dir)
            else:
                print(f"Warning: Directory not found, skipping: {directory}")
        
        if not all_files:
            raise ValueError(f"No .npy files found in provided directories: {dirs_to_scan}")

        # 2. Dynamically discover classes from all filenames.
        all_class_names = set()
        for file_path in all_files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            try:
                class_name = name_without_ext.split('_', 1)[1]
                all_class_names.add(class_name)
            except IndexError:
                continue
        
        sorted_classes = sorted(list(all_class_names))
        self.class_to_idx = {name: i for i, name in enumerate(sorted_classes)}

        # 3. Split the aggregated files into training and validation sets.
        if self.hparams.val_split_ratio > 0.0:
            train_files, val_files = train_test_split(
                all_files, 
                test_size=self.hparams.val_split_ratio, 
                random_state=self.hparams.seed
            )
        else:
            train_files = all_files
            val_files = []
            
        self.train_dataset = IFCFullDataset(train_files, self.class_to_idx)
        self.val_dataset = IFCFullDataset(val_files, self.class_to_idx)
        
        print(f"IFCFullDataModule setup complete:")
        print(f" - Found {len(self.class_to_idx)} classes from {len(all_files)} total files.")
        print(f" - Train samples: {len(self.train_dataset)}")
        print(f" - Validation samples: {len(self.val_dataset)}")


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    @property
    def num_classes(self):
        return len(self.class_to_idx)