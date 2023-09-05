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
from cycling_utils import InterruptableDistributedSampler
from lightning.pytorch.strategies import DDPStrategy


class EpochHandler(L.Callback):

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_end(self, trainer, pl_module):
        self.sampler._reset_progress()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.sampler.advance(len(batch))


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
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    print(args)

    checkpoint_latest = args.save_dir / 'checkpoint_latest.pt'
    checkpoint_latest.parent.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42)

    # Data
    dataset = WikiText2(download=False)
    dist.init_process_group(backend='nccl')

    # Split data in to train, val, test
    n = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n - 4000, 2000, 2000])

    # Create ModelCheckpoint - training run will atomically overwrite args.save_dir / 'checkpoint_latest.pt' every 50 training steps
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, filename='checkpoint_latest', every_n_train_steps=50)

    
    # Instantiate the InterruptibleDistributedSampler that will save your data sampler state
    train_sampler = InterruptableDistributedSampler(train_dataset)

    # Create Sampler callback - it will handle all pre-emption for the ISC
    sampler_callback = EpochHandler(sampler=train_sampler)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=32)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)

    # Model
    model = LanguageModel(vocab_size=dataset.vocab_size)

    # Load previous run state if it exists
    if os.path.exists(checkpoint_latest):
        print(f"Loading checkpoint from {checkpoint_latest}")
        checkpoint = torch.load(checkpoint_latest)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        train_sampler.load_state_dict(checkpoint["sampler_state_dict"])
        completed_epochs = checkpoint["epoch"]

    # Trainer
    # Set checkpoint callback and ensure that the InterruptableDistributedSampler by setting replace_sampler_ddp=False
    trainer = L.Trainer(callbacks=[checkpoint_callback, sampler_callback], gradient_clip_val=0.25, max_epochs=args.epochs, use_distributed_sampler=False, strategy='ddp', accelerator='gpu', devices=6)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":

    main()
