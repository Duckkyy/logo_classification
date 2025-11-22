import argparse
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from model import Model

CLASS_NAMES = ["bad", "good"]

def build_eval_transform(image_size: int = 256):
    """
    Build the evaluation transform used for val/test/inference.
    This should match the eval_transform from your training pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])

def build_model(model_builder, model_type: int) -> nn.Module:
    if model_type == 1:
        return model_builder.build_resnet18()
    elif model_type == 2:
        return model_builder.build_efficientnet_b0()
    elif model_type == 3:
        return model_builder.build_mobilenet_v3()
    elif model_type == 4:
        return model_builder.build_vit_tiny()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from 1, 2, 3, 4.")

def load_image(image_path: str, transform, device: torch.device):
    """
    Load a single image and apply the given transform.
    Returns a tensor of shape [1, C, H, W].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)  # add batch dimension
    return img_t.to(device)

def load_model(model, model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained ResNet-18 model from a .pth / .pt file.
    """
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
):
    """
    Run inference on a single image tensor of shape [1, C, H, W].

    Returns:
        predicted_index (int),
        predicted_label (str),
        probabilities (list[float]) with length = num_classes
    """
    with torch.no_grad():
        outputs = model(image_tensor)          # shape [1, num_classes]
        probs = torch.softmax(outputs, dim=1)  # shape [1, num_classes]
        prob_values, pred_indices = torch.max(probs, dim=1)

        pred_idx = int(pred_indices.item())
        pred_label = CLASS_NAMES[pred_idx]
        prob_list = probs.squeeze(0).cpu().tolist()  # [num_classes]

    return pred_idx, pred_label, prob_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for good/bad defect classification."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.pth or .pt).",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size used for resizing / cropping.",
    )
    parser.add_argument(
        "--model-type", 
        type=int, choices=[1, 2, 3, 4], 
        default=2, 
        help="Model type: 1=ResNet18, 2=EfficientNet-B0, 3=MobileNet-v3, 4=ViT-Tiny"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_builder = Model(num_classes=len(CLASS_NAMES), flag="inference")
    model = build_model(model_builder, model_type=args.model_type)
    if args.model_type == 4:
        args.image_size = 224  # ViT-Tiny requires 224x224 input

    # Select device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Model path : {args.model_path}")
    print(f"Image path : {args.image_path}")

    # Build transform and load image
    transform = build_eval_transform(image_size=args.image_size)
    image_tensor = load_image(args.image_path, transform, device)

    # Load model
    model = load_model(model, args.model_path, device)

    start_time = time.perf_counter()

    # Predict
    pred_idx, pred_label, prob_list = predict_image(model, image_tensor)

    end_time = time.perf_counter()
    print(f"Inference time : {end_time - start_time:.4f} seconds")

    print("\n===== Inference Result =====")
    print(f"Predicted label : {pred_label}")
    print(f"Class index     : {pred_idx}")
    print("Probabilities   :")
    for i, p in enumerate(prob_list):
        print(f"  {CLASS_NAMES[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
