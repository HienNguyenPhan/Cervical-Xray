import torch
from torch import nn
from torchvision.models.detection import MaskRCNN
from src.models.components.ResNet101BackBone import get_resnet101_fpn_backbone

class MyMaskRCNN(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        backbone = get_resnet101_fpn_backbone(pretrained=pretrained_backbone)
        self.model = MaskRCNN(backbone, num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
