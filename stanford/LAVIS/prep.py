from lavis.common.config import Config
import argparse
import lavis.tasks as tasks
from lavis.common.registry import registry
import os
from pathlib import Path
import jdk
import torch
from transformers import BertTokenizer, BertForPreTraining
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=False, default="lavis/projects/blip/train/pretrain_14m.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    print(args)
    return args

cfg = Config(parse_args())
try:
    jdk.install("11")
except jdk.JdkError:
    pass # already installed
Path(cfg.run_cfg.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.run_cfg.tensorboard_path).mkdir(parents=True, exist_ok=True)
coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
# download_url("https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json", coco_gt_root),
# download_url("https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json", coco_gt_root)

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)
c = {'arch': 'blip_pretrain', 'load_pretrained': False, 'pretrained': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth', 'vit_type': 'base', 'vit_grad_ckpt': False, 'vit_ckpt_layer': 0, 'image_size': 224, 'alpha': 0.4, 'med_config_path': 'configs/models/bert_config.json', 'embed_dim': 256, 'prompt': 'a picture of ', 'model_type': 'base', 'queue_size': 57600}
image_encoder = VisionTransformerEncoder.from_config(c, from_pretrained=True)
text_encoder = XBertEncoder.from_config(c, from_pretrained=True)
text_decoder = XBertLMHeadDecoder.from_config(c, from_pretrained=True)
torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu",check_hash=True,)

BertTokenizer.from_pretrained("bert-base-uncased")
BertForPreTraining.from_pretrained("bert-base-uncased")

os.makedirs(".cache/lavis/vg", exist_ok=True)
os.makedirs(".cache/lavis/sbu_captions", exist_ok=True)
os.makedirs(".cache/lavis/conceptual_caption", exist_ok=True)

try:
    os.symlink("/mnt/.node1/Open-Datasets/coco/", ".cache/lavis/coco_gt/images")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/vg/VG_100K/", ".cache/lavis/vg/images")
except FileExistsError:
    pass

with open(".cache/lavis/vg/vg_caption.json", "rt") as fin:
    with open(".cache/lavis/vg/vg_caption.json.out", "wt") as fout:
        for line in fin:
            fout.write(line.replace('/export/share/datasets/vision/visual-genome/image/', ''))
        
os.replace(".cache/lavis/vg/vg_caption.json.out", ".cache/lavis/vg/vg_caption.json")

try:
    os.symlink("/mnt/.node1/Open-Datasets/sbu/dataset/", ".cache/lavis/sbu_captions/images")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/cc/cc3m/", ".cache/lavis/conceptual_caption/images_3m")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/cc/cc12m/", ".cache/lavis/conceptual_caption/images_12m")
except FileExistsError:
    pass