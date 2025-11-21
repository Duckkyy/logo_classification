import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


def collect_image_paths(root_dir):
    """
    Collect all image paths grouped by class.
    Expected dataset structure:
        root_dir/class_name/*.png
    Returns:
        dict: {class_name: [list of image paths]}
    """
    class_to_paths = {}
    root = Path(root_dir)

    for class_dir in root.iterdir():
        if class_dir.is_dir():
            image_paths = []
            for p in class_dir.iterdir():
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(str(p))
            class_to_paths[class_dir.name] = image_paths

    return class_to_paths


def stratified_split(paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Perform stratified split into train/val/test.
    Returns:
        train_paths, val_paths, test_paths
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1."

    random.seed(seed)

    # Create labels for stratify
    all_paths = []
    all_labels = []

    for label, path_list in paths.items():
        all_paths.extend(path_list)
        all_labels.extend([label] * len(path_list))

    # First split: train vs temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=all_labels,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_paths, test_paths, _, _ = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )

    return train_paths, val_paths, test_paths


def copy_images(path_list, output_root):
    """
    Copy images into output folder while preserving class names.
    """
    for img_path in path_list:
        img_path = Path(img_path)
        class_name = img_path.parent.name

        class_output_dir = output_root / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, class_output_dir / img_path.name)


def run_split(
    input_dir="dataset",
    output_dir="dataset_split",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Full pipeline:
    - Read dataset
    - Stratified split
    - Save into dataset_split/train, dataset_split/val, dataset_split/test
    """
    print("\nCollecting dataset...")
    paths = collect_image_paths(input_dir)

    print("Classes found:", list(paths.keys()))
    for cls, p_list in paths.items():
        print(f"  {cls}: {len(p_list)} images")

    print("\nPerforming stratified split...")
    train_paths, val_paths, test_paths = stratified_split(
        paths, train_ratio, val_ratio, test_ratio, seed
    )

    print(f"Train: {len(train_paths)} images")
    print(f"Val:   {len(val_paths)} images")
    print(f"Test:  {len(test_paths)} images")

    # Prepare output folders
    output_root = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    copy_images(train_paths, output_root / "train")
    copy_images(val_paths, output_root / "val")
    copy_images(test_paths, output_root / "test")

    print("\nDone! Split dataset is saved in:", output_root)


if __name__ == "__main__":
    run_split(
        input_dir="./dataset",          # your original dataset path
        output_dir="./dataset_split",   # output folder
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
