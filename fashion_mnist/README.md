# Fashion MNIST
If you've just generated a new Freedom Container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
Then create and source a virtual environment. 
```bash
python3 -m virtualenv ~/.fashion
source ~/.fashion/bin/activate
```
Then clone the `isc-demos` repository and install the `fashion_mnist` requirements to that virtual environment.
```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos/fashion_mnist
pip install -r requirements.txt
```
Then update the Project ID in the experiment launch file `fashion_mnist.isc`
```bash
cd ~/isc-demos/fashion_mnist
nano fashion_mnist.isc
```
with your Project ID from [Control Plane](https://cp.strongcompute.ai/).
```toml
isc_project_id = "<isc-project-id>"
```
Finally, launch an experiment with the following command.
```bash
isc train fashion_mnist.isc
```

## Run in the User Container

```
DIR=/root \
EXPERIMENT_ID=$(python -c 'import uuid; print(uuid.uuid4())') \
LOSSY_ARTIFACT_PATH=$DIR/artifacts/$EXPERIMENT_ID/lossy \
CHECKPOINT_ARTIFACT_PATH=$DIR/artifacts/$EXPERIMENT_ID/checkpoints \
bash -c '\
printenv EXPERIMENT_ID && \
torchrun --nnodes=1 --nproc-per-node=1 train.py --dataset-id 8d2de5b2-d07f-47ce-a6d6-d217a1cfa369 --lr 0.001 --batch-size 16'
```