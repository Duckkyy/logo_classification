import os
import random
from typing import Tuple, List
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from dataset import DataModule
from model import Model


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_weighted_sampler(
    train_dataset,
    train_class_counts: torch.Tensor,
) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    """
    Build a WeightedRandomSampler and class_weights for loss,
    based on the class distribution in the training dataset.

    Args:
        train_dataset: torch.utils.data.Dataset (ImageFolder)
        train_class_counts: tensor of length num_classes, e.g. [700, 245]

    Returns:
        sampler: WeightedRandomSampler for the training DataLoader
        class_weights: tensor of length num_classes, for CrossEntropyLoss
    """
    # class_counts: e.g., tensor([700, 245])
    class_counts = train_class_counts.float()
    num_classes = len(class_counts)

    # Inverse frequency weighting, normalized roughly around 1
    # class_weight[c] ∝ 1 / count[c]
    class_weights = class_counts.sum() / (num_classes * class_counts)

    # Build a per-sample weight list using the class_weights
    sample_weights = []
    for _, label in train_dataset.samples:
        sample_weights.append(class_weights[label])

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_weights


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Single training epoch.
    Returns:
        (epoch_loss, epoch_accuracy)
    """
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluation loop (for val/test).
    Returns:
        (epoch_loss, epoch_accuracy, all_labels, all_preds)
    """
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

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

def train(model=None, root_dir: str = "./dataset", image_size: int = 256, batch_size: int =32, num_workers: int =4, learning_rate_backbone: float =1e-4, learning_rate_head: float =5e-4, weight_decay: float =1e-4, num_epochs: int =20, save_path: str ="best_sampler.pth"):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ================== Data ==================
    data_module = DataModule(
        root_dir=root_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup()

    # Build WeightedRandomSampler and class_weights for loss
    sampler, class_weights = build_weighted_sampler(
        train_dataset=data_module.train_dataset,
        train_class_counts=data_module.train_class_counts,
    )
    print("Class weights (for loss):", class_weights)

    # Train loader uses sampler (no shuffle), val/test standard
    train_loader = data_module.train_dataloader(sampler=sampler)
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    class_names = data_module.class_names
    num_classes = len(class_names)
    print("Class names:", class_names)

    model = model.to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate_backbone},
            {"params": head_params, "lr": learning_rate_head},
        ],
        weight_decay=weight_decay,
    )

    # Use class_weights in CrossEntropyLoss to handle imbalance
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        # verbose=True,
    )

    # ================== Training Loop ==================
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_labels, val_preds = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated at epoch {epoch}. Saved to {save_path}")

    print(f"\nTraining finished. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")

    # ================== Test Evaluation ==================
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")

    print("Classification report (test):")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    print("Confusion matrix (test):")
    print(confusion_matrix(test_labels, test_preds))
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for ResNet18 with weighted sampler.")
    parser.add_argument("--root_dir", type=str, default="./dataset_1", help="Root directory of the dataset.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loaders.")
    parser.add_argument("--learning_rate_backbone", type=float, default=1e-4, help="Learning rate for backbone parameters.")
    parser.add_argument("--learning_rate_head", type=float, default=5e-4, help="Learning rate for head parameters.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--save_path", type=str, default="best_sampler.pth", help="Path to save the best model.")

    args = parser.parse_args()

    model_builder = Model(num_classes=2, flag="train")
    model = model_builder.build_vit_tiny()

    train(
        model=model,
        root_dir=args.root_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate_backbone=args.learning_rate_backbone,
        learning_rate_head=args.learning_rate_head,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
    )
