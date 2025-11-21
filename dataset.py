# dataset.py

import os
import random
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageFolderWithTransform(Dataset):
    """
    Wraps an ImageFolder + a subset of indices + a transform.
    This allows different transforms for train / val / test
    while sharing the same underlying ImageFolder.
    """

    def __init__(self, base_dataset: datasets.ImageFolder,
                 indices: List[int],
                 transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        path, label = self.base_dataset.samples[base_idx]
        image = self.base_dataset.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 256):
    """
    Create transforms for train and eval (val/test).
    Train has augmentation, val/test only resizing + normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def stratified_split_indices(
    targets: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified split indices (train/val/test) based on class labels.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1."

    set_seed(seed)

    num_samples = len(targets)
    num_classes = len(set(targets))

    # Collect indices for each class
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for c, idx_list in class_to_indices.items():
        random.shuffle(idx_list)
        n = len(idx_list)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # rest

        train_indices.extend(idx_list[:n_train])
        val_indices.extend(idx_list[n_train:n_train + n_val])
        test_indices.extend(idx_list[n_train + n_val:])

    # Shuffle each split (optional but nice)
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def create_dataloaders(
    data_dir: str,
    image_size: int = 256,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Main function to create:
      - train_loader, val_loader, test_loader
      - class_names (['bad', 'good'] etc.)
      - train_class_counts (tensor of length num_classes)
    Assumes data_dir has the structure:
        data_dir/good/*.png
        data_dir/bad/*.png
    """

    set_seed(seed)

    # Base dataset without transform
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx
    print("Classes:", class_names)
    print("Class to index:", class_to_idx)

    # Get all labels
    targets = [s[1] for s in base_dataset.samples]

    # Split indices stratified
    train_idx, val_idx, test_idx = stratified_split_indices(
        targets,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(f"#Train: {len(train_idx)}, #Val: {len(val_idx)}, #Test: {len(test_idx)}")

    # Count classes in train split (for weight calculation)
    num_classes = len(class_names)
    train_class_counts = torch.zeros(num_classes, dtype=torch.long)
    for i in train_idx:
        label = targets[i]
        train_class_counts[label] += 1
    print("Train class counts:", train_class_counts.tolist())

    # Transforms
    train_transform, eval_transform = get_transforms(image_size=image_size)

    # Datasets with their own transform
    train_dataset = ImageFolderWithTransform(
        base_dataset=base_dataset,
        indices=train_idx,
        transform=train_transform,
    )

    val_dataset = ImageFolderWithTransform(
        base_dataset=base_dataset,
        indices=val_idx,
        transform=eval_transform,
    )

    test_dataset = ImageFolderWithTransform(
        base_dataset=base_dataset,
        indices=test_idx,
        transform=eval_transform,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names, train_class_counts


if __name__ == "__main__":
    """
    Quick test:
    Run `python dataset.py` to check that dataloaders work.
    """
    data_dir = "./dataset"  # change to your dataset path

    train_loader, val_loader, test_loader, class_names, train_class_counts = \
        create_dataloaders(
            data_dir=data_dir,
            image_size=256,
            batch_size=32,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

    print("Class names:", class_names)
    print("Train class counts:", train_class_counts)
    batch = next(iter(train_loader))
    images, labels = batch
    print("One train batch shape:", images.shape, labels.shape)
