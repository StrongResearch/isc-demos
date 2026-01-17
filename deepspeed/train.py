import argparse
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, disable_progress_bars
from cycling_utils import AtomicDirectory
disable_progress_bars()

print("Hello from script land")

def parse_args():
    parser = argparse.ArgumentParser()
    # deepspeed requires these
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)

    # args specifically for this project
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_tag", type=str, default=None)
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=None
    )

    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # prepare dataset
    train_data_path = os.path.join(args.data_path, "train-00000-of-00001.parquet")
    test_data_path = os.path.join(args.data_path, "test-00000-of-00001.parquet")
    dataset = load_dataset("parquet", data_files={"train": train_data_path, "test": test_data_path}, cache_dir="/tmp/wiki_qa")

    print(f"dataset['train'] examples:")
    for example in dataset["train"][0:5]:
        print(example)

    def preprocess_function(examples):
        # Combine question and answer into a single text
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal language modeling (shift input_ids right)
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Process dataset
    train_dataset = dataset["train"].map(
        preprocess_function, batched=True,
        remove_columns=dataset["train"].column_names, desc="Tokenizing and preprocessing train dataset"
    )

    # test_dataset = dataset["test"].map(
    #     preprocess_function, batched=True,
    #     remove_columns=dataset["test"].column_names, desc="Tokenizing and preprocessing test dataset"
    # )

    print(f"train_dataset examples:")
    for example in train_dataset[0:5]:
        print(example)

    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }

    print("Dataset and dataloader done")

    # OPTIMIZER INITIALIZED BY DEEPSPEED
    # # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("Optimizer done")

    # SCHEDULER INITIALIZED BY DEEPSPEED?
    # # Scheduler
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=100, num_training_steps=5000
    # )

    print("Scheduler done")

    # DeepSpeed initialization
    model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        collate_fn=collate_fn,
        # OPTIMIZER INITIALIZED BY DEEPSPEED
        # optimizer=optimizer,
        # SCHEDULER INITIALIZED BY DEEPSPEED
        # lr_scheduler=scheduler,
        config=args.deepspeed_config,
        model_parameters=model.parameters(),
    )

    # print(f"DeepSpeed model_engine init: {model_engine}")
    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    is_master = deepspeed.comm.get_rank() == 0
    saver = AtomicDirectory(output_directory=output_directory, is_master=is_master)

    global_step = 0

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        print(f"Resuming from AtomicDirector {latest_checkpoint_path}")
        load_path, client_state = model_engine.load_checkpoint(args.output_dir, args.resume_tag)
        global_step = client_state["global_step"]
        print(f"Resumed from DeepSpeed checkpoint {load_path} at step {global_step}")

    for epoch in range(2):
        for batch in train_dataloader:

            # Move batch to device
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            outputs = model_engine(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)

            loss = outputs.loss
            print(f"Batch {global_step} loss {loss.item()}")

            model_engine.backward(loss)
            model_engine.step()

            global_step += 1

            checkpoint_directory = saver.prepare_checkpoint_directory()

            tag = f"step_{global_step}"
            model_engine.save_checkpoint(checkpoint_directory, tag, client_state={"global_step": global_step}, save_latest=True)
            
            saver.symlink_latest(checkpoint_directory)

            print(f"Checkpoint saved at {checkpoint_directory}")


if __name__ == "__main__":
    main()
