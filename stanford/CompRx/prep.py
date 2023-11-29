import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch_fidelity import FeatureExtractorInceptionV3
from clip import load
from huggingface_hub import hf_hub_download

vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
FeatureExtractorInceptionV3("a", ["logits"])
load("ViT-B/32", "cpu")

hf_hub_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 'open_clip_pytorch_model.bin', revision=None, cache_dir=None)