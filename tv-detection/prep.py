import torchvision
from torchvision.models import resnet50, ResNet50_Weights

# weights = torchvision.models.get_weight('ResNet50_Weights.IMAGENET1K_V1')
_ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# weights = torchvision.models.get_weight(...)

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
_  = resnet_fpn_backbone(backbone_name="resnet101", weights="ResNet101_Weights.IMAGENET1K_V1")