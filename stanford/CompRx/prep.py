import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch_fidelity import FeatureExtractorInceptionV3

vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
FeatureExtractorInceptionV3("a", ["logits"])