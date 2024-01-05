# MaskRCNN

First create and source a virtual environment for this project.

```bash
cd ~
python3 -m virtualenv ~/.mask
source ~/.mask/bin/activate
```

Install the necessary requirements for this project.

```bash
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
python prep.py
```

Finally, launch the experiment on the ISC.

```bash
cd isc-demos/maskrcnn
isc train maskrcnn_resnet101_fpn.isc
```