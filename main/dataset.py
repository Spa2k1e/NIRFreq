import os
import random
from typing import Dict, Tuple, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from scipy import io
from sorcery import dict_of
from torch.utils.data import Dataset, DataLoader


def _blur_down(
    img: torch.Tensor,
    scale: float = 0.25,
    ksize: Tuple[int, int] = (3, 3),
    sigma: Tuple[float, float] = (1.5, 1.5),
) -> torch.Tensor:
    """
    Applies Gaussian blur and then downsamples the image.

    Args:
        img (torch.Tensor): The input image tensor.
        scale (float): The scale factor for downsampling.
        ksize (Tuple[int, int]): The kernel size for the Gaussian blur.
        sigma (Tuple[float, float]): The sigma for the Gaussian blur.

    Returns:
        torch.Tensor: The processed, downsampled image tensor.
    """
    blur = gaussian_blur2d(img, ksize, sigma)
    return F.interpolate(blur, scale_factor=scale, mode="bicubic", align_corners=False)


class NBUDataset(Dataset):
    """
    A dataset class for loading NBU pansharpening data, which is typically in .mat format.
    It handles data splitting, normalization, and preprocessing according to Wald's protocol.
    """

    def __init__(
        self,
        data_dir: str,
        sensor: str,
        split: str = "train",
        ori_test: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initializes the NBUDataset.

        Args:
            data_dir (str): The root directory of the dataset.
            sensor (str): The sensor type (e.g., 'gf', 'ikonos', 'wv2').
            split (str): The dataset split, one of ["train", "val", "test"].
            ori_test (bool): If True, uses the original full-resolution data for testing.
            train_ratio (float): The proportion of data to use for training.
            val_ratio (float): The proportion of data to use for validation.
            random_seed (int): The seed for shuffling the dataset.
            verbose (bool): If True, prints information about the dataset.
        """
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.split = split
        self.ori_test = ori_test

        mat_dir = os.path.join(self.data_dir, "MS_256")
        if not os.path.isdir(mat_dir):
            raise FileNotFoundError(f"MS_256 directory not found in {self.data_dir}")
        
        mat_files = sorted(os.listdir(mat_dir))
        num_files = len(mat_files)

        # Ensure reproducible splits
        random.seed(random_seed)
        random_sequence = list(range(num_files))
        random.shuffle(random_sequence)

        train_end_idx = int(num_files * train_ratio)
        val_end_idx = train_end_idx + int(num_files * val_ratio)

        if self.split == "train":
            self.mat_files = [mat_files[i] for i in random_sequence[:train_end_idx]]
        elif self.split == "val":
            self.mat_files = [mat_files[i] for i in random_sequence[train_end_idx:val_end_idx]]
        elif self.split == "test":
            self.mat_files = [mat_files[i] for i in random_sequence[val_end_idx:]]
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of ['train', 'val', 'test'].")

        # TODO: Move sensor-specific configurations to a dedicated config file.
        if sensor.lower() == 'gf':
            self.max_val = 1023.0
        elif sensor.lower() in ['ikonos', 'wv2', 'wv3', 'quickbird']:
            self.max_val = 2047.0
        else:
            raise ValueError(f"Unsupported sensor type: {sensor}")

        if verbose:
            print(f"Initialized {self.split} split with {len(self.mat_files)} samples. First 5: {self.mat_files[:5]}")

    def __len__(self) -> int:
        return len(self.mat_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single data sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            A dictionary containing 'ms', 'pan', 'up_ms', and 'gt' tensors.
        """
        ms_path = os.path.join(self.data_dir, "MS_256", self.mat_files[idx])
        pan_path = os.path.join(self.data_dir, "PAN_1024", self.mat_files[idx])

        try:
            mat_ms = io.loadmat(ms_path)
            mat_pan = io.loadmat(pan_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found at index {idx}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading .mat file at index {idx}: {e}")

        key_ms = "imgMS" if "imgMS" in mat_ms else "I_MS"
        key_pan = "imgPAN" if "imgPAN" in mat_pan else "I_PAN"
        if key_pan not in mat_pan:
            key_pan = "block" # Handle alternative key names

        if key_ms not in mat_ms or key_pan not in mat_pan:
            raise KeyError(f"Could not find expected keys in .mat files for index {idx}.")

        ms = torch.from_numpy((mat_ms[key_ms] / self.max_val).astype(np.float32)).permute(2, 0, 1)
        pan = torch.from_numpy((mat_pan[key_pan] / self.max_val).astype(np.float32)).unsqueeze(0)
        gt = ms.clone() # Ground truth is the original high-res MS

        # Wald's protocol: blur and downsample to create low-res inputs
        if self.split in ["train", "val"] or not self.ori_test:
            ms = _blur_down(ms.unsqueeze(0)).squeeze(0)
            pan = _blur_down(pan.unsqueeze(0)).squeeze(0)

        up_ms = F.interpolate(ms.unsqueeze(0), size=(pan.shape[-2], pan.shape[-1]), mode="bicubic", align_corners=False).squeeze(0)

        filename = self.mat_files[idx]
        return dict_of(ms, pan, up_ms, gt, filename)

class plNBUDataset(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the NBU dataset.
    It encapsulates all the data loading and preparation logic, providing separate
    dataloaders for training, validation, and testing.
    """
    def __init__(
        self,
        data_dir: str,
        sensor: str,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        test_mode: str = 'reduced', # New argument
        **kwargs, # Absorb unused arguments like save_visuals
    ):
        """
        Initializes the plNBUDataset DataModule.

        Args:
            data_dir (str): The root directory of the dataset.
            sensor (str): The sensor type.
            batch_size (int): The batch size for the training dataloader.
            num_workers (int): The number of workers for data loading.
            pin_memory (bool): If True, pins memory for faster GPU transfer.
            test_mode (str): 'reduced' or 'full' to select the test dataset.
        """
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Assigns train/val/test datasets. Called on 1 GPU/TPU in Trainer.
        """
        if stage == 'fit' or stage is None:
            self.dataset_train = NBUDataset(self.hparams.data_dir, sensor=self.hparams.sensor, split="train")
            self.dataset_val = NBUDataset(self.hparams.data_dir, sensor=self.hparams.sensor, split="val")
        if stage == 'test' or stage is None:
            if self.hparams.test_mode == 'reduced':
                self.dataset_test = NBUDataset(self.hparams.data_dir, sensor=self.hparams.sensor, split="test", ori_test=False)
            elif self.hparams.test_mode == 'full':
                self.dataset_test = NBUDataset(self.hparams.data_dir, sensor=self.hparams.sensor, split="test", ori_test=True)

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(
            self.dataset_val,
            batch_size=1, # Typically use batch size 1 for validation
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the appropriate test dataloader based on test_mode."""
        return DataLoader(
            self.dataset_test,
            batch_size=1, # Batch size is 1 for testing
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
