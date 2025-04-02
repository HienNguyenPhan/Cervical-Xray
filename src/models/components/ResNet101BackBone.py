import torch
from torch import nn

class ResNet101Backbone(nn.Module):
    """ResNet-101 backbone for feature extraction in Mask R-CNN."""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        return [x]