# FashionMNIST
## Step 1a: `isc-demos` image container
If you have created your container based on the `isc-demos` Image in [Control Plane](https://cp.strongcompute.ai/), 
then your container already has all necessary dependencies installed including a python virtual environment with 
necessary python dependencies at `/opt/venv`. Please clone this `isc-demos` repository and then skip to Step 2.

```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```

## Step 1b: other image container
If you've just created a new container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
Then create and source a virtual environment. 
```bash
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
```
Then clone the `isc-demos` repository and install the `fashion_mnist` requirements to that virtual environment.
```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos/fashion_mnist
pip install -r requirements.txt
```

## Step 2: prepare and launch experiment
Update the Project ID in the experiment launch file `fashion_mnist.isc`
```bash
cd ~/isc-demos/fashion_mnist
nano fashion_mnist.isc
```
with your Project ID from [Control Plane](https://cp.strongcompute.ai/).
```toml
isc_project_id = "<isc-project-id>"
```
Also notice that the experiment launch file is configured to launch a `burst` experiment.
```toml
compute_mode = "burst"
```
Burst experiments run on clusters which are created by the ISC specifically for those experiments and then destroyed when the experiment is terminated.

Stage a `burst` experiment with the following command.
```bash
isc train fashion_mnist.isc
```
Your terminal will report `[notice] filesystem set to READ-ONLY` to indicate that your container has been locked while an image of your container 
is created for running the experiment. This experiment image will include the edits you just made to your `fashion_mnist.isc` file.

After the experiment image has finished being created your terminal will report `[notice] filesystem set to READ/WRITE` and the experiment image will 
be exported to cloud storage.

Return to Control Plane > Experiments to launch the `burst` experiment by clicking "Burst" when the experiment is ready to launch. The "Burst"
button will become active after the experiment image has finished exporting to cloud storage (this can take several minutes depending
on the size of the image).


