import argparse
import os
import shutil
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, disable_progress_bars
from cycling_utils import TimestampedTimer, AtomicDirectory

timer = TimestampedTimer("Hello from train.py")
disable_progress_bars()


def parse_args():
    parser = argparse.ArgumentParser()
    # deepspeed requires these
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser = deepspeed.add_config_arguments(parser)

    # args specifically for this project
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=None)

    # LoRA config
    lora_config = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # prepare dataset
    train_data_path = os.path.join(args.data_path, "train-00000-of-00001.parquet")
    test_data_path = os.path.join(args.data_path, "test-00000-of-00001.parquet")
    dataset = load_dataset(
        "parquet", data_files={"train": train_data_path, "test": test_data_path}, cache_dir="/tmp/wiki_qa"
    )

    def preprocess_function(examples):
        # Combine question and answer into a single text
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]

        # Tokenize with padding and truncation
        encodings = tokenizer(texts, truncation=True, max_length=512, padding="max_length", return_tensors=None)

        # Create labels for causal language modeling (shift input_ids right)
        encodings["labels"] = encodings["input_ids"].copy()

        return encodings

    # Process dataset
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and preprocessing train dataset",
    )

    print(f"Rank {os.environ['RANK']} training example:\n{train_dataset[0]}")

    # Create dataloader
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
            "labels": torch.stack([torch.tensor(x["labels"]) for x in batch]),
        }

    # Note: optimizer and learning rate scheduler are initialized by deepspeed, see
    # ds_config.json for configuration details.

    # DeepSpeed initialization
    model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        collate_fn=collate_fn,
        config=args.deepspeed_config,
        model_parameters=model.parameters(),
    )

    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    is_master = deepspeed.comm.get_rank() == 0
    saver = AtomicDirectory(output_directory=output_directory, is_master=is_master)

    # tracking a global_step parameter to demonstrate use of the
    # deepspeed checkpointing "client_state" utility.
    global_step = 0

    # detect if there is a checkpoint to resume from
    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)

        timer.report(f"Resuming from AtomicDirector {latest_checkpoint_path}")
        load_path, client_state = model_engine.load_checkpoint(latest_checkpoint_path)
        global_step = client_state["global_step"]

        timer.report(f"Resumed from DeepSpeed checkpoint {load_path} at step {global_step}")

    for epoch in range(5):
        for batch in train_dataloader:

            # Move batch to device
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            # forward pass
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # loss
            loss = outputs.loss
            timer.report(f"Batch {global_step} loss {loss.item()}")

            # backward pass
            model_engine.backward(loss)

            # optimizer step
            model_engine.step()

            # increment global_step
            global_step += 1

            # Strong Compute prepares checkpoint directory
            checkpoint_directory = saver.prepare_checkpoint_directory()

            # manual tagging not strictly necessary
            tag = f"step_{global_step}"

            # By specifying save_latest=True deepspeed will save a "latest" file to the
            # checkpoint_directory which deepspeed will then use to "detect" the latest
            # checkpoint to resume from. A new "latest" file will be saved with each checkpoint
            # artifact.
            model_engine.save_checkpoint(
                checkpoint_directory, tag, client_state={"global_step": global_step}, save_latest=True
            )

            # Strong Compute finalizes checkpoint
            saver.symlink_latest(checkpoint_directory)

            timer.report(f"Checkpoint saved at {checkpoint_directory}")


if __name__ == "__main__":
    main()
