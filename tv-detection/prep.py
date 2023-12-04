# Downloading ResNet101 backbone weights ahead of training
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
_  = resnet_fpn_backbone(backbone_name="resnet101", weights="ResNet101_Weights.IMAGENET1K_V1")
_  = resnet_fpn_backbone(backbone_name="resnet50", weights="ResNet50_Weights.IMAGENET1K_V1")