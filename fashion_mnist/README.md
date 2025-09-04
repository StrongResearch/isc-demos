# Fashion MNIST
If you've just created a new container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
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
Also notice that the experiment launch file is configured to launch a `burst` experiment.
```toml
compute_mode = "burst"
```
Burst experiments run on clusters which are created by the ISC specifically for those experiments and then destroyed when the experiment is terminated.

Stage a `burst` experiment with the following command.
```bash
isc train fashion_mnist.isc
```
Return to Control Plane > Experiments to launch the `burst` experiment by clicking "Burst" when the experiment is ready to launch.
