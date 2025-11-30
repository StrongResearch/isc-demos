import argparse
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--deepspeed", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_tag", type=str, default=None)
    return p.parse_args()


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
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
    text = ["Hello world!"] * 2048
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=5000
    )

    # DeepSpeed initialization
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=args.deepspeed,
        model_parameters=model.parameters(),
    )

    global_step = 0

    # Resume logic
    if args.resume_tag:
        ckpt_path = os.path.join(args.output_dir, args.resume_tag)
        print(f"Resuming from {ckpt_path}")
        model_engine.load_checkpoint(args.output_dir, args.resume_tag)
        global_step = load_custom_metadata(ckpt_path)
        print(f"Resumed at step {global_step}")

    for epoch in range(2):
        for batch in dataloader:
            batch = [t.to(model_engine.local_rank) for t in batch]
            input_ids, attention_mask = batch

            outputs = model_engine(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=input_ids)

            loss = outputs.loss
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
