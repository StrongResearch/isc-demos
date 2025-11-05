# MaskRCNN
## Step 1a: `isc-demos` image container
If you have created your container based on the `isc-demos` Image in [Control Plane](https://cp.strongcompute.ai/), 
then your container already has all necessary dependencies installed including a python virtual environment with 
necessary python dependencies at `/opt/venv`. Please clone this `isc-demos` repository and then skip to Step 2.

```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```

## Step 1b: other image container
First create and source a virtual environment for this project.
```bash
cd ~
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
```

Install the necessary requirements for this project.

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
cd isc-demos/maskrcnn
pip install -r requirements.txt
```

Also ensure cycling_utils is cloned and installed.

```bash
cd ~
git clone https://github.com/StrongResearch/cycling_utils.git
pip install -e cycling_utils
```

Run "prep.py" to download pretrained model weights before launching your training job.

```bash
cd isc-demos/maskrcnn
python prep.py
```

Finally, launch the experiment on the ISC.

```bash
cd isc-demos/maskrcnn
isc train maskrcnn_resnet101_fpn.isc
```

Results viewed on tensorboard should resemble the following output.

![maskrcnn_tensorboard](https://github.com/StrongResearch/isc-demos/blob/adam-maskimages/maskrcnn/tensorboard.png)

Results of experiments conducted to test the scaling properties of Mask RCNN are presented on the following charts.

![maskrcnn_acc_vs_time](https://github.com/StrongResearch/isc-demos/blob/adam-maskimages/maskrcnn/acc_vs_time.png)

![maskrcnn_speedup_vs_time](https://github.com/StrongResearch/isc-demos/blob/adam-maskimages/maskrcnn/speedup_vs_time.png)
