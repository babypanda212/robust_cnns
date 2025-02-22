# src/models/resnet18.py
import torch
from torch.nn import Conv2d
from torchvision.models import resnet18

def get_resnet18(num_classes=10, pretrained=False):
    """
    Returns a modified ResNet18 model suitable for CIFAR-10.
    """
    model = resnet18(pretrained=pretrained)
    model.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Disable max pooling
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
