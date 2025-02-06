# Fine-tune DeepSeek-R1
## Quick start
### Step 1: Prepare your container
If you've just generated a new Freedom Container, start by installing python.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
### Step 2: Create a virtual environment
Next create and source a `.deepseek` python virtual environment.
```bash
cd ~
python3 -m virtualenv ~/.deepseek
source ~/.deepseek/bin/activate
```
### Step 3: Install dependencies
Clone this repo and install dependencies.
```bash
ch /root
git clone https://github.com/StrongResearch/isc-demos.git
cd /root/isc-demos/deepseek
pip install -r requirements.txt
```
### Step 4: Update experiment launch file
Pick your DeekSeek-R1 model. We have imported these as Datasets in Control Plane.
Edit the experiment launch file (`deepseek-r1-<model>.isc`) corresponding to the model of your choice, replacing the `project_id` with your Project ID.
```toml
isc_project_id = "<project-id>"
experiment_name = "deepseek-r1-<model>"
gpus = 16
compute_mode = "cycle"
output_path = "~/outputs/deepseek-r1-<model>"
dataset_id = "<dataset-id>"
command = '''...'''
```
### Step 5: Launch your experiment to train
```bash
isc train deepseek-r1-<model>.isc
isc experiments
```
