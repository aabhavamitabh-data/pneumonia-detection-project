# Defines two models:
# 1. SimpleCNN - a basic convolutional network built from scratch (baseline)
# 2. get_resnet_model - a pretrained ResNet18 fine-tuned for our 3 classes

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A basic CNN built from scratch, used as our baseline model."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # squash to [batch, 256, 1, 1]
            nn.Flatten(),
            nn.Dropout(0.5),           # helps prevent overfitting on small data
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet_model(num_classes=3, freeze_backbone=True):
    """
    Loads a ResNet18 pretrained on ImageNet and replaces its final
    layer to output our 3 classes. This is transfer learning - we
    reuse features the model already learned from millions of images,
    which matters a lot given how small our dataset is (317 images).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze all pretrained layers so only the new final layer trains.
        # Good starting point for a small dataset - prevents overfitting.
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer to match our 3 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Note: model.fc is created fresh, so its params require_grad=True by default

    return model


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through both models
    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2 fake images

    baseline = SimpleCNN(num_classes=3)
    out1 = baseline(dummy_input)
    print("SimpleCNN output shape:", out1.shape)  # expect [2, 3]

    resnet = get_resnet_model(num_classes=3)
    out2 = resnet(dummy_input)
    print("ResNet18 output shape:", out2.shape)  # expect [2, 3]