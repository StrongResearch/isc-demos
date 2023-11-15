"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from cycling_utils import atomic_torch_save
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized, main_process
from lavis.datasets.builders import load_dataset
from lavis.common.logger import MetricLogger, SmoothedValue
from torch.utils.data import DataLoader
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            if name == "train":
                datasets["train"] = load_dataset("coco_caption", vis_path="/mnt/.node1/Open-Datasets/coco/train2014")
                continue
            if name == "valid":
                datasets["valid"] = load_dataset("coco_caption", vis_path="/mnt/.node1/Open-Datasets/coco/val2014")
                continue
            if name == "test":
                datasets["test"] = load_dataset("coco_caption", vis_path="/mnt/.node1/Open-Datasets/coco/test2014")
                continue
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError
    
    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True, writer=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        start_iters=1,
        args=None,
        writer=None,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            start_iters=start_iters,
            args=args,
            writer=writer,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        args=None,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            args=args,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        args=None,
        writer=None,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        
        header = "Train: data epoch: [{}]".format(epoch)
        
        metric_stuff = []

        for i in metric_logger.log_every(range(iters_per_epoch), 1, header, start_iters):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            lr_scheduler.step(cur_epoch=epoch, cur_step=start_iters - 1) #TODO change step?

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (start_iters + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if isinstance(data_loader, DataLoader):
                sampler = data_loader.sampler
            else:
                sampler = data_loader._dataloader.sampler
            sampler.advance(args.run_cfg.batch_size_train)

            logging.info("Synchronizing...")
            metric_logger.synchronize_between_processes()
            if writer is not None:
                bunch = []
                for metre, value in metric_logger.meters.items():
                    if metre == "loss_lm":
                        new_string = ""
                        in_bracket = False
                        for char in str(value):
                            if char == "(":
                                in_bracket = True
                                continue
                            elif char == ")":
                                break
                            if in_bracket:
                                new_string += char
                                
                        value = new_string
                    bunch.append(("Train/" + str(metre), float(str(value)), start_iters))
                    #writer.add_scalar("Train/" + str(metre), float(str(value)), start_iters)
                for tup in bunch:
                    metric_stuff.append(tup)
                    
            if is_main_process():
                if start_iters >= len(data_loader):
                    logging.info(f"Reached the end of the dataloder; Saving checkpoint at iters: {start_iters}")
                    self.save_checkpoint(model, optimizer, sampler, args, scaler, epoch + 1, 1)
                    if writer is not None:
                        for scalar in metric_stuff:
                            writer.add_scalar(scalar[0], scalar[1], scalar[2])
                        metric_stuff = []
                elif start_iters % args.run_cfg.get("checkpoint_freq", 100) == 0:
                    logging.info(f"Saving checkpoint at iters: {start_iters} and epoch: {epoch}")
                    self.save_checkpoint(model, optimizer, sampler, args, scaler, epoch, start_iters)
                    if writer is not None:
                        for scalar in metric_stuff:
                            writer.add_scalar(scalar[0], scalar[1], scalar[2])
                        metric_stuff = []
                    logging.info("Averaged stats: " + str(metric_logger.global_avg()))
                    
            dist.barrier()
            start_iters += 1

        # after train_epoch()
        # gather the stats from all processes
        logging.info("Synchronizing...")
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        if is_main_process():
            logging.info(metric_logger)
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    
    def save_checkpoint(self, model, optimizer, train_sampler, config, scaler, epoch, iteration, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        
        model_no_ddp = self.unwrap_dist_model(model, config)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "train_sampler" : train_sampler.state_dict(),
            "config": config.to_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "iteration": iteration,
        }
        save_to = os.path.join(
            config.run_cfg.output_dir,
            "{}_{}.pth".format(self.name, "best" if is_best else "latest"),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_to))
        atomic_torch_save(save_obj, save_to)
        logging.info("Saved successfully")
    
    def unwrap_dist_model(self, model, config):
        use_dist = config.run_cfg.distributed
        if use_dist:
            return model.module
        else:
            return model

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file