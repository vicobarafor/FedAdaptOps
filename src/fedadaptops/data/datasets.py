from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10(data_dir: str | Path) -> tuple[Dataset, Dataset]:
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tfms)
    test = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_tfms)
    return train, test


def make_loader(
    dataset: Dataset,
    indices: list[int] | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
