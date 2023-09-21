import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from cycling_utils import InterruptableDistributedSampler, EpochHandler
from lightning.pytorch.strategies import DDPStrategy


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    print(args)

    checkpoint_latest = args.save_dir
    checkpoint_latest.parent.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42)

    # Data
    dataset = WikiText2(download=False)
    dist.init_process_group(backend='nccl')

    # Split data in to train, val, test
    n = len(dataset)
    train_dataset, val_dataset = random_split(dataset, [n - 2000, 2000])

    # Create ModelCheckpoint - training run will atomically overwrite args.save_dir / 'checkpoint_latest.pt' every 50 training steps
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, every_n_train_steps=args.save_freq, save_last=True)

    # Instantiate the InterruptibleDistributedSampler that will save your data sampler state
    train_sampler = InterruptableDistributedSampler(train_dataset)

    # Create Sampler callback - it will handle all pre-emption for the ISC
    sampler_callback = EpochHandler(sampler=train_sampler, checkpoint_dir=args.save_dir, save_freq=args.save_freq)

    # Load previous run state if it exists
    if os.path.exists(checkpoint_latest / 'sampler_last.pt'):
        sampler_checkpoint = torch.load(checkpoint_latest / 'sampler_last.pt')
        train_sampler.load_state_dict(sampler_checkpoint["sampler_state_dict"])
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=32)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)

    # Model
    model = LanguageModel(vocab_size=dataset.vocab_size)

    num_gpus = torch.cuda.device_count()
    num_nodes = dist.get_world_size() / num_gpus

    trainer = L.Trainer(callbacks=[checkpoint_callback, sampler_callback], log_every_n_steps=1, gradient_clip_val=0.25, max_epochs=args.epochs, use_distributed_sampler=False, strategy='ddp', accelerator='gpu', devices=num_gpus, num_nodes=num_nodes)

    # Trainer
    # Set checkpoint callback and ensure that the InterruptableDistributedSampler by setting replace_sampler_ddp=False
    if os.path.exists(checkpoint_latest / 'last.ckpt'):

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_latest / 'last.ckpt')
    else:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    exit()

if __name__ == "__main__":

    main()
