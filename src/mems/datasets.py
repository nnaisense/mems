import os

import numpy as np
import pytorch_lightning as pl
import torch
from mems.utils import download_file
from omegaconf import DictConfig
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

INPUT_SHAPES = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "ffhq256": (3, 256, 256),
}


def get_data_module(cfg: DictConfig) -> pl.LightningDataModule:
    if cfg.dataset.lower() == "mnist":
        dm = MNISTDataModule(cfg)
    elif cfg.dataset.lower() == "cifar10":
        dm = CIFAR10DataModule(
            data_dir=cfg.data_dir,
            val_split=cfg.val_split,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            shuffle=cfg.shuffle,
            drop_last=True,
            pin_memory=True,
        )
        dm.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        dm.val_transforms = transforms.ToTensor()
        dm.test_transforms = transforms.ToTensor()
    elif cfg.dataset.lower() == "ffhq256":
        dm = FFHQ256DataModule(
            data_dir=cfg.data_dir,
            val_split=cfg.val_split,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            shuffle=cfg.shuffle,
        )
        dm.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        dm.val_transforms = transforms.ToTensor()
        dm.test_transforms = transforms.ToTensor()
    else:
        raise NotImplementedError(
            f"Invalid data.dataset: {cfg.dataset}. " f'Must be in {"mnist", "cifar10", "ffhq256"}.'
        )
    return dm


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        MNIST(root=self.cfg.data_dir, train=True, download=True)

    def setup(self, stage):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if stage == "fit":
            self.mnist_train = MNIST(root=self.cfg.data_dir, train=True, transform=transform)

        if stage == "test":
            self.mnist_test = MNIST(root=self.cfg.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.cfg.batch_size, shuffle=False)


class FFHQ256DataModule(LightningDataModule):

    name = "ffhq256"
    file_url = "https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq-256.npy"

    def __init__(
        self,
        data_dir: str,
        val_split: int = 7000,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (3, 256, 256)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.data_dir = data_dir if data_dir is not None else os.getcwd()

    @property
    def file_path(self):
        return os.path.expanduser(os.path.join(self.data_dir, "FFHQ256", "ffhq-256.npy"))

    def prepare_data(self):
        if os.path.exists(self.file_path):
            return

        base_dir = os.path.dirname(self.file_path)
        if not os.path.exists(base_dir):
            print(f"Creating {base_dir}")
            os.makedirs(base_dir)

        print("Downloading " + self.file_url + " to " + self.file_path + "... This will take about an hour.")
        download_file(self.file_url, self.file_path)
        print("Download complete.")

    def train_dataloader(self):
        dataset = np.load(self.file_path, mmap_mode="r")
        dataset = TensorDataset(torch.as_tensor(dataset), torch.zeros(dataset.shape[0]))
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        dataset = np.load(self.file_path, mmap_mode="r")
        dataset = TensorDataset(torch.as_tensor(dataset))
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader

    def test_dataloader(self):
        """Currently the test data is the same as val data, as used by VD-VAE."""
        dataset = np.load(self.file_path, mmap_mode="r")
        dataset = TensorDataset(torch.as_tensor(dataset))
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader
