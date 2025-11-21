import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataModule:
    def __init__(
        self,
        root_dir: str = "./dataset",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = os.path.join(root_dir, "train")
        self.val_dir = os.path.join(root_dir, "val")
        self.test_dir = os.path.join(root_dir, "test")

        self.train_transform, self.eval_transform = self._build_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.class_names = None
        self.train_class_counts = None

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Build train and eval (val/test) transforms.

        NOTE:
            - Train transform includes augmentation.
            - Eval transform is clean (only resize + normalize).
        """
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
            ),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return train_transform, eval_transform

    def setup(self):
        """
        Create ImageFolder datasets for train/val/test and
        compute class information for the training split.
        """
        # Train dataset with augmentation
        self.train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform,
        )

        # Val and test datasets with eval transform
        self.val_dataset = datasets.ImageFolder(
            root=self.val_dir,
            transform=self.eval_transform,
        )

        self.test_dataset = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.eval_transform,
        )

        # Class names (e.g., ['bad', 'good'] or ['good', 'bad'])
        self.class_names = self.train_dataset.classes
        print("Classes:", self.class_names)
        print("Class to index:", self.train_dataset.class_to_idx)

        # Count how many samples of each class in the training split
        num_classes = len(self.class_names)
        self.train_class_counts = torch.zeros(num_classes, dtype=torch.long)

        # train_dataset.samples = list of (path, label)
        for _, label in self.train_dataset.samples:
            self.train_class_counts[label] += 1

        print("Train class counts:", self.train_class_counts.tolist())

    def train_dataloader(self, sampler: Optional[torch.utils.data.Sampler] = None) -> DataLoader:
        """
        Build DataLoader for training.

        Args:
            sampler: optional custom sampler (e.g., WeightedRandomSampler).
                     If sampler is provided, shuffle is automatically disabled.

        Returns:
            DataLoader for training data.
        """
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Build DataLoader for validation data."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Build DataLoader for test data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# if __name__ == "__main__":
#     # Quick sanity check
#     data_module = GoodBadDataModule(
#         root_dir="./dataset_split",
#         image_size=256,
#         batch_size=32,
#         num_workers=4,
#     )

#     data_module.setup()

#     train_loader = data_module.train_dataloader()
#     val_loader = data_module.val_dataloader()
#     test_loader = data_module.test_dataloader()

#     # Fetch one batch to check shapes
#     images, labels = next(iter(train_loader))
#     print("Train batch image shape:", images.shape)
#     print("Train batch labels shape:", labels.shape)
