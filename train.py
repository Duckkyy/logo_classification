import os
import random
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_transforms(image_size: int = 256) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    image_size: int = 256,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Create train and validation dataloaders using ImageFolder.

    Returns:
        train_loader, val_loader, class_names, class_counts
    """
    train_transform, val_transform = get_data_transforms(image_size=image_size)

    # Load the whole dataset with train transform first (we will override for val subset)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes
    print("Classes:", class_names)         # e.g. ['bad', 'good'] or ['good', 'bad']
    print("Class to index mapping:", full_dataset.class_to_idx)

    # Compute class counts (for information and class weights)
    targets = [sample[1] for sample in full_dataset.samples]
    num_classes = len(class_names)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    for t in targets:
        class_counts[t] += 1
    print("Class counts:", class_counts.tolist())

    # Split into train and validation
    num_samples = len(full_dataset)
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Set different transforms for validation dataset
    # random_split returns Subset, so we override the transform attribute
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names, class_counts


def create_resnet18_model(num_classes: int = 2) -> nn.Module:
    """Load ResNet-18 pretrained on ImageNet and replace the final layer."""
    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc, all_labels, all_preds


def main():
    # ===================== Config =====================
    data_dir = "./dataset"          # Change this to your dataset path
    batch_size = 32
    num_epochs = 20
    image_size = 256
    val_ratio = 0.2
    learning_rate_backbone = 1e-4
    learning_rate_head = 5e-4
    seed = 42

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===================== Data =====================
    train_loader, val_loader, class_names, class_counts = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        image_size=image_size,
    )

    num_classes = len(class_names)

    # ===================== Model =====================
    model = create_resnet18_model(num_classes=num_classes)
    model = model.to(device)

    # Fine-tune all layers (do not freeze anything)
    # If you wanted to freeze backbone, you would set requires_grad = False here.
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": learning_rate_backbone},
        {"params": head_params, "lr": learning_rate_head},
    ])

    # Class weights for imbalance: inverse proportional to class frequency
    class_counts = class_counts.float()
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        # verbose=True
    )

    # ===================== Training loop =====================
    best_val_acc = 0.0
    best_model_path = "best_resnet18_good_bad.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, all_labels, all_preds = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated. Saved to {best_model_path}")

        # Optional: print classification report every few epochs
        if epoch == num_epochs:
            print("\nClassification report on validation set:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
            print("Confusion matrix:")
            print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
