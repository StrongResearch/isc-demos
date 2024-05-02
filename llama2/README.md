# Introduction
This demonstration fine-tunes a Llama2-7B model via Parameter Efficient Fine Tuning (PEFT) using Low Rank Adaptation (LoRA), and Fully Sharded Data Parallel (FSDP) to manage GPU memory. We demonstrate implementation of the "FULL" sharding strategy which shards the model and optimizer accross all GPUs within the cluster.

**Note:** The llama-recipes GitHub repository by Meta Research was cloned on 21 November 2023 for the purpose of developing this demonstration, and is not updated with any subsequent changes to the llama-recipes GitHub repository.

# Getting started
## 1. Obtain the model
For this demonstration, you will need a copy of the Llama2 7B model downloaded from Huggingface. This demonstration uses the `Llama-2-7b-hf` variant: https://huggingface.co/meta-llama/Llama-2-7b-hf.

Once downloaded, make a note of the path to the pre-trained model, for example `<local_path>/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/<snapshot_id>`.

## 2. Configure a suitable virtual environment
FSDP currently relies on Pytorch nighlies, so we recommend creating and sourcing a new virtual environment for this demonstration.

```bash
python3 -m virtualenv ~/.llama2
source ~/.llama2/bin/activate
```

Install the required version of Pytorch with the following command.

```bash
# pip install torch==2.2.0.dev20231113+cu118 --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Navigate into the llama-recipes directory and install llama-recipes.

```bash
cd ~/isc-demos/llama2/llama-recipes
pip install -e .
```

Ensure `cycling_utils` from StrongCompute is also installed (first clone from `https://github.com/StrongResearch/cycling_utils.git`)

```bash
cd ~/cycling_utils
pip install -e .
```

Also install tensorboard.

```bash
cd ~
pip install tensorboard==2.15.1
```

### 2.a Optionally adjust `peft` to avoid noisy warnings
When resuming a peft model from pause, we encountered the following warning: 
```bash
UserWarning: for base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight: copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op. (Did you mean to pass `assign=True` to assign items in the state dictionary to their corresponding key in the module instead of copying them in place?)
```
We found that this can be avoided by modifying the `set_peft_model_state_dict` function in the `peft` package file `.../site-packages/peft/utils/save_and_load.py` at line 158 as follows.
```bash
FROM: load_result = model.load_state_dict(peft_model_state_dict, strict=False)
TO: load_result = model.load_state_dict(peft_model_state_dict, strict=False, assign=True)
```

## 3. Pre-process the dataset
Run the following commands to pre-process the training dataset.

```bash
cd ~/isc-demos/llama2/llama-recipes
python prep_data.py --model-path <path-to-your-model>
```

## 4. Update the ISC file
Open the ISC file at `~/isc-demos/llama2/llama-recipes/llama2.isc` and update the `--model_name` parameter with `<path-to-your-model>`.

## 5. Launch the experiment on the ISC
Launch the experiment on the ISC with the following command.

```bash
cd ~/isc-demos/llama2/llama-recipes
isc train llama2.isc
```