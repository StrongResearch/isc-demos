from lavis.common.config import Config
import argparse
import lavis.tasks as tasks

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
task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)