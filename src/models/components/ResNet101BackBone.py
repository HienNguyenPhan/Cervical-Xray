import torch
from torch import nn

class ResNet101Backbone(nn.Module):
    """ResNet-101 backbone for feature extraction in Mask R-CNN."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(weights=None)
        self.stage1 = nn.Sequential(*list(resnet.children())[:4])
        self.stage2 = nn.Sequential(*list(resnet.children())[4])
        self.stage3 = nn.Sequential(*list(resnet.children())[5])
        self.stage4 = nn.Sequential(*list(resnet.children())[6])
        self.stage5 = nn.Sequential(*list(resnet.children())[7])

    def forward(self, x):
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c2, c3, c4, c5]