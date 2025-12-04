import argparse
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, disable_progress_bars
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


def save_custom_metadata(save_dir, global_step):
    """Only rank 0 saves small extra metadata."""
    if deepspeed.comm.get_rank() == 0:
        torch.save(
            {"global_step": global_step},
            os.path.join(save_dir, "training_state.pt")
        )


def load_custom_metadata(load_dir):
    path = os.path.join(load_dir, "training_state.pt")
    if os.path.exists(path):
        return torch.load(path)["global_step"]
    return 0


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Dummy dataset
    # text = ["Hello world!"] * 2048
    # enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # prepare dataset
    train_data_path = os.path.join(args.data_path, "train-00000-of-00001.parquet")
    test_data_path = os.path.join(args.data_path, "test-00000-of-00001.parquet")
    dataset = load_dataset("parquet", data_files={"train": train_data_path, "test": test_data_path}, cache_dir="/tmp/wiki_qa")

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

    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }
    
    # train_sampler = InterruptableDistributedSampler(tokenized_train_dataset)
    # test_sampler = InterruptableDistributedSampler(tokenized_test_dataset)

    # train_dataloader = DataLoader(tokenized_train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler)
    # test_dataloader = DataLoader(tokenized_test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=test_sampler)

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
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        collate_fn=collate_fn,
        # OPTIMIZER INITIALIZED BY DEEPSPEED
        # optimizer=optimizer,
        # SCHEDULER INITIALIZED BY DEEPSPEED?
        # lr_scheduler=scheduler,
        config=args.deepspeed_config,
        model_parameters=model.parameters(),
    )

    print("DeepSpeed model_engine init")

    global_step = 0

    # Resume logic
    if args.resume_tag:
        ckpt_path = os.path.join(args.output_dir, args.resume_tag)
        print(f"Resuming from {ckpt_path}")
        model_engine.load_checkpoint(args.output_dir, args.resume_tag)
        global_step = load_custom_metadata(ckpt_path)
        print(f"Resumed at step {global_step}")

    for epoch in range(2):
        for batch in train_dataloader:

            batch = [t.to(model_engine.local_rank) for t in batch]
            input_ids, attention_mask = batch

            outputs = model_engine(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=input_ids)

            loss = outputs.loss
            print(f"Batch {global_step} loss {loss.item()}")

            model_engine.backward(loss)
            model_engine.step()

            global_step += 1

            # Save checkpoints periodically on all ranks
            if global_step % 500 == 0 and deepspeed.comm.get_rank() == 0:
                tag = f"step_{global_step}"
                ckpt_dir = os.path.join(args.output_dir, tag)
                os.makedirs(ckpt_dir, exist_ok=True)

                model_engine.save_checkpoint(args.output_dir, tag)
                save_custom_metadata(ckpt_dir, global_step)
                print(f"Checkpoint saved at {ckpt_dir}")



if __name__ == "__main__":
    main()
