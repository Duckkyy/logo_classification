import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

class Model:
    def __init__(self, num_classes: int, flag="train"):
        self.num_classes = num_classes
        self.flag = flag

    def build_resnet18(self):
        if self.flag == "train":
            weight = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weight = None
        model = models.resnet18(weights=weight)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model
    
    def build_efficientnet_b0(self):
        if self.flag == "train":
            weight = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weight = None
        model = models.efficientnet_b0(weights=weight)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model
    
    def build_vit_tiny(self):
        if self.flag == "train":
            pretrained = True
        else:
            pretrained = False

        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=self.num_classes,
        )
        return model