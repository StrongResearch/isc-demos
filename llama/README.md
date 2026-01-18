# PEFT (LoRA) Fine-tune Llama3.2
Here we demonstrate a minimum-viable fine-tuning of the Llama3.2 models (adaptable for any similar LLM) using FSDP and LoRA.

## Step 1a: `isc-demos` image container
If you have created your container based on the `isc-demos` Image in [Control Plane](https://cp.strongcompute.ai/), 
then your container already has all necessary dependencies installed including a python virtual environment with 
necessary python dependencies at `/opt/venv`. Please clone this `isc-demos` repository and then skip to Step 2.

```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```

## Step 1b: other image container
If you've just generated a new container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
Next create and source a python virtual environment.
```bash
cd /root
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
```
Clone this repo and install dependencies.
```bash
cd /root
git clone https://github.com/StrongResearch/isc-demos.git
cd /root/isc-demos/llama
pip install -r requirements.txt
```

### Step 2: Update experiment launch file
Pick your Llama3.2 model. We have imported these as Datasets in [Control Plane](https://cp.strongcompute.ai/).
Edit the experiment launch file (`llama_3.2_<model>.isc`) corresponding to the model of your choice, replacing the `project_id` with your Project ID.
```toml
isc_project_id = "<project-id>"
experiment_name = "Llama3.2 <model>"
gpus = 16
compute_mode = "cycle"
dataset_id_list = ["dataset-id-0", "dataset-id-1"]
command = '''...'''
```
### Step 3: Launch your experiment to train
```bash
isc train llama_3.2_<model>.isc
isc experiments
```

## Step 4: Debugging / Optimising Checkpoints

If you're training bigger models (such as the 70b), you may find that training with compute_mode = "cycle" does not make enough progress to checkpoint your model.

In this case, you should still train with compute_mode = "cycle" to confirm your training code is valid on a smaller model (e.g. 1B) and then launch in burst to see checkpoints
changes.

## Full fine-tuning
To adapt the `fsdp.py` training scrip to do a full fine-tune rather than PEFT (LoRA) fine-tune, comment out lines 84 to 94 of `fsdp.py`.
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
