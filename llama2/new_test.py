from cycling_utils import TimestampedTimer

timer = TimestampedTimer()

import torch, os
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.utils import fsdp_auto_wrap_policy

from llama_recipes.utils.train_utils import (
    # train,
    # train_epoch,
    # freeze_transformer_layers,
    # setup,
    # setup_environ_flags,
    # clear_gpu_cache,
    # print_model_size,
    get_policies
)

from peft import (
    LoraConfig,
    get_peft_model,
    # get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    # set_peft_model_state_dict,
    # PrefixTuningConfig,
    # TaskType
)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
timer.report(f"local rank {local_rank}, rank {rank}, world size {world_size}")

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()

train_config.enable_fsdp = True
train_config.low_cpu_fsdp = True
train_config.use_peft = True
train_config.peft_method = 'lora'
train_config.use_fp16 = True 
train_config.batch_size_training = 1
fsdp_config.use_fp16 = True 
fsdp_config.pure_bf16 = True
checkpoint_type = StateDictType.FULL_STATE_DICT

model_name = "/mnt/Client/Adamstn3rh22tykvgyhdkclook3rnk7q/adaadam4qalumfvjdstjpx7zyvlebh2u/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
use_cache = False if train_config.enable_fsdp else None

## -- BUILDING THE MODEL -- ##

if rank == 0:
    print("Hello from rank 0")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        use_cache=use_cache,
    )
    timer.report("constructed model (on cpu)")
else:
    llama_config = LlamaConfig.from_pretrained(train_config.model_name)
    llama_config.use_cache = use_cache
    with torch.device("meta"):
        model = LlamaForCausalLM(llama_config)

timer.report(f"loading model from_pretrained")

tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

timer.report(f"loading tokenizer")

## -- INSERTING PEFT -- ##

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.to(torch.bfloat16)

timer.report(f"converting to PEFT model")

## -- WRAPPING IN FSDP -- ##

my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
mixed_precision_policy, _ = get_policies(fsdp_config, rank)

model = FSDP(
    model,
    auto_wrap_policy= my_auto_wrapping_policy,
    cpu_offload=None,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
    sync_module_states=train_config.low_cpu_fsdp,
    param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    if train_config.low_cpu_fsdp and rank != 0 else None
)

timer.report(f"wrapping model with FSDP")

optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
)

timer.report(f"created optimizer")

def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    if rank == 0:
        torch.save(optim_state, full_save_path)
