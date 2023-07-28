# ISC Demos

Demo training runs for the ISC

## Installation

```bash
# install demos
git clone https://github.com/StrongCompute/isc-demos.git
cd isc-demos
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# install launcher
git clone https://github.com/StrongCompute/strong_launcher.git
cd strong_launcher
pip install -e .
cd ..
```

## Usage

Look at `Makefile` for commands to run.

e.g.
```bash
make do_tiny_training
```