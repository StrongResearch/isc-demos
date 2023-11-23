Allow print from other ranks (dist_utils.py)
Changed logger.py to allow start from a certain iteration for when a checkpoint is loaded mid-epoch (stanford/LAVIS/lavis/common/logger.py)
Changed cache root (stanford/LAVIS/lavis/configs/default.yaml)
removed instance ids adding (stanford/LAVIS/lavis/datasets/datasets/base_dataset.py) + (stanford/LAVIS/lavis/datasets/datasets/coco_caption_datasets.py)
altered hyper-params for coco caption (stanford/LAVIS/lavis/projects/blip/train/caption_coco_ft.yaml)

(runner_base.py)
Added eval_freq argument
added tensorboard and a tensorboard path argument in caption_coco_ft.yaml
Loads checkpoint from the path specified in the yaml
changed distributedsampler to InterruptableDistributedSampler
Changed the train/eval loop to use the interruptabledistributedsampler context
Iteration is saved in the checkpoint and resumed from in the train loop
Pass args + iters + tb writer into train function
pass writer to eval

(base_task.py)
changed the loop to use the checkpointed iteration count for when the loop has been interrupted mid-epoch
Checkpoints every 100 iters or whatever the checkpoint_freq config option is set to
tensorboard metrics are recorded on loss and learning rate

added a name variable to captioning.py (captioning.py)

updated requirements

added a prep.sh script that should be run before training to pre-load files that would waste time on the isc