# Installation Notes:
From isc-demos/detectron: 
```
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements
cd detectron2
pip install -e .
``````
# Config notes:
Change config.yml to desired checkpoint/tb/result paths
Change run.isc to desired venv/output

# Output notes:
Output formatted in detectron2/detectron2/utils/events.py
Additional output files disabled in detectron2/detectron2/utils/logger.py
