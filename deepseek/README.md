# PEFT (LoRA) Fine-tune DeepSeek-R1
Here we demonstrate a minimum-viable fine-tuning of the DeepSeek-R1 models (adaptable for any similar LLM) using FSDP and LoRA.
## Quick start
### Step 1: Prepare your container
If you've just generated a new Freedom Container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
### Step 2: Create a virtual environment
Next create and source a `.deepseek` python virtual environment.
```bash
cd /root
python3 -m virtualenv /root/.deepseek
source /root/.deepseek/bin/activate
```
### Step 3: Install dependencies
Clone this repo and install dependencies.
```bash
cd /root
git clone https://github.com/StrongResearch/isc-demos.git
cd /root/isc-demos/deepseek
pip install -r requirements.txt
```
### Step 4: Update experiment launch file
Pick your DeekSeek-R1 model. We have imported these as Datasets in [Control Plane](https://cp.strongcompute.ai/).
Edit the experiment launch file (`deepseek-r1-<model>.isc`) corresponding to the model of your choice, replacing the `project_id` with your Project ID.
```toml
isc_project_id = "<project-id>"
experiment_name = "deepseek-r1-<model>"
gpus = 16
compute_mode = "cycle"
dataset_id_list = ["<dataset-id>"]
command = '''...'''
```
### Step 5: Launch your experiment to train
```bash
isc train deepseek-r1-<model>.isc
isc experiments
```

## Step 6: Debugging / Optimising Checkpoints

If you're training bigger models (such as the 70b), you may find that training with compute_mode = "cycle" does not make enough progress to checkpoint your model.

In this case, you should still train with compute_mode = "cycle" to confirm your training code is valid on a smaller model (e.g. 1.5B) and then launch in burst to see checkpoints
changes.

## Full fine-tuning
To adapt the `fsdp.py` training scrip to do a full fine-tune rather than PEFT (LoRA) fine-tune, comment out lines 83 to 93 of `fsdp.py`.
```
#    # inject PEFT modules
#    lora_config = LoraConfig(
#        r=16,
#        lora_alpha=32,
#        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#        lora_dropout=0, # set to zero to see identical loss on all ranks
#    )
#
#    model = LoraModel(model, lora_config, ADAPTER_NAME)
#
#    timer.report(f"PEFT model: {count_trainable_parameters(model)}")
```
