from lavis.common.config import Config
import argparse
import lavis.tasks as tasks
from lavis.common.registry import registry
from torchvision.datasets.utils import download_url
import os
from pathlib import Path
import jdk

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=False, default="lavis/projects/blip/train/caption_coco_ft.yaml", help="path to configuration file.")
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
jdk.install("11")
Path(cfg.run_cfg.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.run_cfg.tensorboard_path).mkdir(parents=True, exist_ok=True)
coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
download_url("https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json", coco_gt_root),
download_url("https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json", coco_gt_root)

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)
