import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from fsdp_utils import AppState

adapter_name = "ExampleLora"

tokenizer = AutoTokenizer.from_pretrained("/data")
model = AutoModelForCausalLM.from_pretrained(
    "/data", 
    use_cache=False, 
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0, # set to zero to see identical loss on all ranks
)

model = LoraModel(model, lora_config, adapter_name).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state_dict = { "app": AppState(model, optimizer)}
dcp.load(state_dict=state_dict, checkpoint_id="/root/fsdp_backup/CHK2")

prompt = "Do you think there are more wheels or doors in the world?"

# https://arxiv.org/abs/2501.12948
deepseek_r1_input = f'''
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {prompt}. Assistant:'''

encoding = tokenizer(deepseek_r1_input, return_tensors="pt")

input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")

generate_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100, do_sample=True, temperature=0.8)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(answer[0])
