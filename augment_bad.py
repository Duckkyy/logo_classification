import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms


def set_seed(seed: int = 42):
    random.seed(seed)


def get_image_paths(folder: str):
    """Return a list of all image paths in the given folder."""
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    folder = Path(folder)
    paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return paths


def build_augmentation_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.2,
            hue=0.05
        ),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
    ])


def offline_augment_bad(
    bad_dir: str,
    target_count: int = 800,
    image_size: int = 256,
    seed: int = 42
):
    set_seed(seed)

    bad_dir = Path(bad_dir)
    bad_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load existing images
    image_paths = get_image_paths(bad_dir)
    current_count = len(image_paths)

    print(f"Current bad images : {current_count}")
    print(f"Target bad images  : {target_count}")

    if current_count >= target_count:
        print("Already have enough images. Nothing to do.")
        return

    # Step 2: Build augmenter
    aug_transform = build_augmentation_transform(image_size=image_size)

    # Step 3: Augmentation loop
    to_generate = target_count - current_count
    print(f"Need to generate   : {to_generate} augmented images\n")

    # start index for naming new files
    base_index = 0

    # avoid name conflicts
    while True:
        candidate = bad_dir / f"bad_aug_{base_index:05d}.png"
        if not candidate.exists():
            break
        base_index += 1

    for i in range(to_generate):
        src_path = random.choice(image_paths)
        img = Image.open(src_path).convert("RGB")

        # apply augmentation
        aug_img = aug_transform(img)

        # save augmented image
        save_name = bad_dir / f"bad_aug_{base_index:05d}.png"
        aug_img.save(save_name)
        base_index += 1

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{to_generate} images")

    print("\nOffline augmentation completed.")
    print(f"Final number of bad images: {len(get_image_paths(bad_dir))}")


if __name__ == "__main__":
    bad_train_dir = "./dataset/train/bad"

    offline_augment_bad(
        bad_dir=bad_train_dir,
        target_count=800,  
        image_size=256,
        seed=42
    )
