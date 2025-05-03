from torchvision.models.detection import MaskRCNN
from src.models.components.ResNet101BackBone import get_resnet101_fpn_backbone

def get_maskrcnn_with_backbone(pretrained=True, num_classes=7):
    backbone = get_resnet101_fpn_backbone(pretrained=pretrained)
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model
