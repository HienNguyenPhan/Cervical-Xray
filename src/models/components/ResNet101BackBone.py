from torchvision import models
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import torch.nn as nn

def get_resnet101_fpn_backbone(pretrained=True, out_channels=256):
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)

    # Remove fully connected layers
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )

    return_layers = {
        '4': '0',  # resnet.layer1
        '5': '1',  # resnet.layer2
        '6': '2',  # resnet.layer3
        '7': '3',  # resnet.layer4
    }

    in_channels_list = [256, 512, 1024, 2048]

    fpn_backbone = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    return fpn_backbone
