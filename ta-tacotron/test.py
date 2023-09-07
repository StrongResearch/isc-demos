from safetensors.torch import save_model
from model.tacotron2 import SyncedTacotron2 
    
model = SyncedTacotron2()
print(model.state_dict())
save_model(model, "model.sf")