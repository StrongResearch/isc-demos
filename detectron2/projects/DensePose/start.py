from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
  default_argument_parser, default_setup, default_writers
)
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    verify_results
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage
from detectron2.utils import comm

from densepose import add_densepose_config
from densepose.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from densepose.engine import Trainer
from densepose.evaluation.evaluator import DensePoseCOCOEvaluator
from densepose.modeling.densepose_checkpoint import DensePoseCheckpointer

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import itertools
import logging
import os
from collections import OrderedDict

def setup(args):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="densepose")
    return cfg


def main():
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    # disable strict kwargs checking: allow one to specify path handle
    # hints through kwargs, like timeout in DP evaluation
    PathManager.set_strict_kwargs_checking(False)
    
    dist.init_process_group(backend = "nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    
    model = build_model(cfg)
    model = DDP(model, device_ids=[device_id])
    train(cfg, model)
    return

def train(cfg, device_id, model):
    max_iter = cfg.SOLVER.MAX_ITER
    
    writers = default_writers(
    cfg.OUTPUT_DIR + "/../tensorboard/", max_iter
    ) if comm.is_main_process() else []
    
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
    checkpointer = DensePoseCheckpointer(model,
                                                cfg.OUTPUT_DIR,
                                                optimizer=optimizer, 
                                                scheduler=scheduler)
    
    resume = os.path.isfile(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'))
    start_iter = checkpointer.resume_or_load(
        cfg.OUTPUT_DIR, resume=resume
    ).get('iteration', -1) + 1
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, 
                                                 cfg.SOLVER.CHECKPOINT_PERIOD, 
                                                 max_iter=max_iter)

    with EventStorage(start_iter) as storage:
        for data, iter in zip(itertools.islice(
            data_loader, start_iter, max_iter), range(start_iter, max_iter)):
            
            storage.iter = iter
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            
            loss_dict_reduced = {k: v.item() for k,
            v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            storage.put_scalar('lr', optimizer.param_groups[0]['lr'], smoothing_hint=False)
            scheduler.step()
            
            for writer in writers:
                writer.write()
            
            if cfg.TEST.EVAL_PERIOD > 0 and \
                iter + 1 % cfg.TEST.EVAL_PERIOD == 0 and \
                iter != max_iter - 1:
                test(cfg, model)
                comm.synchronize()
            
            periodic_checkpointer.step(iter)

    return

def test(cfg, model):
  results = OrderedDict()
  for dataset_name in cfg.DATASETS.TEST:
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = DensePoseCOCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    result = inference_on_dataset(model, data_loader, evaluator)
    results[dataset_name] = result
  if len(results) == 1:
    results = list(results.values())[0]
  return results


if __name__ == "__main__":
    main()
