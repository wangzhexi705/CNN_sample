import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import CLASSES, TrainConfig


class STL10FolderDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def _collect_samples(self):
        samples = []
        for class_name in CLASSES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for image_path in sorted(class_dir.glob("*.png")):
                samples.append((image_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(config: TrainConfig, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(config.image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4467, 0.4398, 0.4066),
                    std=(0.2603, 0.2566, 0.2713),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4467, 0.4398, 0.4066),
                std=(0.2603, 0.2566, 0.2713),
            ),
        ]
    )


def split_train_valid_indices(dataset_size: int, val_ratio: float, seed: int):
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    valid_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, valid_indices


def create_dataloaders(config: TrainConfig):
    train_dataset = STL10FolderDataset(
        config.data_dir / "train",
        transform=build_transforms(config, train=True),
    )
    valid_dataset = STL10FolderDataset(
        config.data_dir / "train",
        transform=build_transforms(config, train=False),
    )
    train_indices, valid_indices = split_train_valid_indices(
        len(train_dataset),
        config.val_ratio,
        config.seed,
    )
    train_set = Subset(train_dataset, train_indices)
    valid_set = Subset(valid_dataset, valid_indices)
    if config.max_train_samples is not None:
        train_set = Subset(train_dataset, train_indices[: config.max_train_samples])
    if config.max_valid_samples is not None:
        valid_set = Subset(valid_dataset, valid_indices[: config.max_valid_samples])

    test_set = STL10FolderDataset(
        config.data_dir / "test",
        transform=build_transforms(config, train=False),
    )
    if config.max_test_samples is not None:
        test_set = Subset(test_set, list(range(config.max_test_samples)))

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, test_loader
