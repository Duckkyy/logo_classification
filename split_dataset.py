import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)


def collect_image_paths(root_dir: str) -> Dict[str, List[Path]]:
    """
    Scan a dataset directory and collect all image paths per class.

    Returns:
        {"good": [Path1, Path2], "bad": [...], ...}
    """
    root = Path(root_dir)
    class_to_paths = {}

    for class_dir in root.iterdir():
        if class_dir.is_dir():
            images = [p for p in class_dir.iterdir()
                      if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
            class_to_paths[class_dir.name] = images

    return class_to_paths


def stratified_split(
    paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split a single class list into train/val/test using given ratios.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.shuffle(paths)
    n = len(paths)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test gets the remainder
    n_test = n - n_train - n_val

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


def copy_files(paths: List[Path], dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for src in paths:
        shutil.copy(src, dst_dir)


def split_dataset(
    src_dataset: str = "./dataset",
    dst_dataset: str = "./dataset_split",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Perform a stratified split into train/val/test and copy images.
    """
    set_seed(seed)

    # Step 1: Collect image paths
    class_to_paths = collect_image_paths(src_dataset)
    print("Found classes:", list(class_to_paths.keys()))

    # For storing all split results
    split_data = {
        "train": {},
        "val": {},
        "test": {}
    }

    # Step 2: Split each class
    for class_name, paths in class_to_paths.items():
        print(f"\nSplitting class '{class_name}' ({len(paths)} images)...")

        train_paths, val_paths, test_paths = stratified_split(
            paths, train_ratio, val_ratio, test_ratio
        )

        split_data["train"][class_name] = train_paths
        split_data["val"][class_name] = val_paths
        split_data["test"][class_name] = test_paths

        print(f"  train: {len(train_paths)}")
        print(f"  val  : {len(val_paths)}")
        print(f"  test : {len(test_paths)}")

    # Step 3: Copy files into new structure
    for split_name, classes in split_data.items():
        for class_name, paths in classes.items():
            dst_dir = f"{dst_dataset}/{split_name}/{class_name}"
            copy_files(paths, dst_dir)

    print("\nDataset successfully split!")
    print(f"Output saved to: {dst_dataset}")


if __name__ == "__main__":
    split_dataset(
        src_dataset="./dataset_raw", 
        dst_dataset="./dataset", 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
