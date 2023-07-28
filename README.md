# ISC Demos

Demo training runs for the ISC

## Quickstart

```bash
# install demos
cd ~
python3 -m virtualenv ~/.venv
source ~/.venv/bin/activate

git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos
pip install -r requirements.txt

# hello world
cd ~/isc-demos/tiny-training
isc train config.isc

# maskrcnn detection
cd ~/isc-demos/tv-detection
isc train config.isc

# for nerf
cd ~/isc-demos/nerf_ddp
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
isc train config.isc
```

```bash
isc experiments # view a list of your experiments
isc clusters # view the status of the clusters
```