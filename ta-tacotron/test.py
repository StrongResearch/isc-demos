import io
import os
import pathlib
import shutil
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file

t = torch.randn(5,5)
# Your io.BytesIO buffer

destination_file = pathlib.Path("test.safetensors")

save = True

if save:
    with tempfile.NamedTemporaryFile() as temp_file:
        save_file({"tensor":t},temp_file.name)
        print("saving")
        os.rename(temp_file.name, destination_file)

d = load_file(destination_file)
print(d["tensor"])
