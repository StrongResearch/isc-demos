'''
@file main.py
@author strongcompute-henry
@date 2023-10-09
@brief detectron2 demo - implementation is based on /tools/plain_train_net.py
and uses the mask rcnn r50 fpn 3x model for training instance segmentation.
cycling is also implemented to run on the isc
'''

debug: bool = False

import os
import datetime
import argparse
import itertools
import logging
import yaml
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
  build_detection_train_loader, build_detection_test_loader
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
  default_argument_parser, default_setup, default_writers
)
from detectron2.evaluation import (
  COCOEvaluator,
  inference_on_dataset,
  print_csv_format
)
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

def load_params(file_path: str) -> Dict[str, Any]:
  with open(file_path, 'r') as file:
    params = yaml.safe_load(file)
  return params

def setup_dist_print(is_master: bool) -> None:
  builtin_print = __builtins__.print
  def print(*args, **kwargs):
    if is_master:
      builtin_print(*args, **kwargs)
  __builtins__.print = print

def init_dist_mode(params: Dict[str, Any]) -> None:
  dist.init_process_group('nccl')
  comm.create_local_process_group(params['NUM_WORKERS'])
  dist.barrier()
  setup_dist_print(comm.get_rank() == 0)

def setup(params: Dict[str, Any], args: argparse.ArgumentParser):
  # init dataset
  register_coco_instances(
    'train', {}, params['TRAIN_ANNOTATION_PATH'], params['TRAIN_DATASET_PATH']
  )
  register_coco_instances(
    'test', {}, params['TEST_ANNOTATION_PATH'], params['TEST_DATASET_PATH']
  )

  # setup config file for coco - instance segmentation training
  cfg = get_cfg()
  default_setup(cfg, args)

  cfg.merge_from_file(model_zoo.get_config_file(params['CONFIG_FILE']))
  cfg.DATASETS.TRAIN = ('train',)
  cfg.DATASETS.TEST = ('test',)
  cfg.MODEL.MASK_ON = params['MASK_ON']
  cfg.MODEL.RESNETS.DEPTH = params['RESNETS_DEPTH']
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params['BATCH_SIZE_PER_IMAGE']
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['CONFIG_FILE'])
  cfg.DATALOADER.NUM_WORKERS = params['NUM_WORKERS']
  cfg.SOLVER.CHECKPOINT_PERIOD = params['CHECKPOINT_PERIOD']
  cfg.SOLVER.IMS_PER_BATCH = params['IMS_PER_BATCH']
  cfg.SOLVER.BASE_LR = params['BASE_LR']
  cfg.SOLVER.MAX_ITER = params['MAX_ITER']
  cfg.SOLVER.STEPS = params['STEPS']
  cfg.freeze()
  return cfg

def test(params: Dict[str, Any], cfg, model) -> None:
  results = OrderedDict()
  for dataset_name in cfg.DATASETS.TEST:
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = COCOEvaluator(dataset_name, output_dir=params['RESULT_PATH'])
    result = inference_on_dataset(model, data_loader, evaluator)
    results[dataset_name] = result
    if comm.is_main_process():
      logging.info(f'evaluation results for {dataset_name} in csv format:')
      print_csv_format(result)
  if len(results) == 1:
    results = list(results.values())[0]
  return results

def train(params: Dict[str, Any], cfg, model, resume=False) -> None:
  model.train()
  optimiser = build_optimizer(cfg, model)
  scheduler = build_lr_scheduler(cfg, optimiser)

  checkpointer = DetectionCheckpointer(
    model, params['CHECKPOINT_PATH'], optimizer=optimiser, scheduler=scheduler
  )

  resume = os.path.isfile(
    os.path.join(params['CHECKPOINT_PATH'], 'last_checkpoint')
  )
  start_iter = checkpointer.resume_or_load(
    cfg.MODEL.WEIGHTS, resume=resume
  ).get('iteration', -1) + 1
  max_iter = cfg.SOLVER.MAX_ITER

  periodic_checkpointer = PeriodicCheckpointer(
    checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
  )

  writers = default_writers(
    params['TENSORBOARD_PATH'], max_iter
  ) if comm.is_main_process() else []

  data_loader = build_detection_train_loader(cfg)

  if comm.is_main_process():
    logging.info(f'starting training from iteration {start_iter}')

  with EventStorage(start_iter) as storage:
    for data, iteration in zip(itertools.islice(
      data_loader, start_iter, max_iter), range(start_iter, max_iter)):
      storage.iter = iteration

      if debug:
        breakpoint()

      loss_dict = model(data)
      losses = sum(loss_dict.values())

      # ensure loss values are valid i.e. no exploding/vanishing gradient
      assert torch.isfinite(losses).all(), loss_dict

      loss_dict_reduced = {k: v.item() for k,
        v in comm.reduce_dict(loss_dict).items()}
      losses_reduced = sum(loss for loss in loss_dict_reduced.values())
      if comm.is_main_process():
        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

      optimiser.zero_grad()
      losses.backward()
      optimiser.step()
      storage.put_scalar(
        'lr', optimiser.param_groups[0]['lr'], smoothing_hint=False
      )
      scheduler.step()

      if cfg.TEST.EVAL_PERIOD > 0 and \
        iteration+1 % cfg.TEST.EVAL_PERIOD == 0 and \
        iteration != max_iter-1:
        test(params, cfg, model)
        comm.synchronize()

      if iteration-start_iter > 5 and \
      (iteration+1)%params['ITER_LOG_PERIOD'] == 0 or \
      iteration == max_iter-1:
        for writer in writers:
          writer.write()
      periodic_checkpointer.step(iteration)

def main():
  logging.basicConfig(level=logging.DEBUG)

  args = default_argument_parser().parse_args()
  params = load_params('config.yml')

  os.makedirs(params['CHECKPOINT_PATH'], exist_ok=True)
  os.makedirs(params['TENSORBOARD_PATH'], exist_ok=True)
  os.makedirs(params['RESULT_PATH'], exist_ok=True)


  init_dist_mode(params)
  cfg = setup(params, args)

  model = build_model(cfg)
  model = DistributedDataParallel(
    model, device_ids=[comm.get_rank() % torch.cuda.device_count()]
  )

  train(params, cfg, model, resume=args.resume)
  test(params, cfg, model)

if __name__ == '__main__':
  print(f"{datetime.datetime.now()} STARTING:")
  main()