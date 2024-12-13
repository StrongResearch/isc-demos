import os
import glob
from ultralytics import YOLO
import torch

# Load the model (change here if you want another model)
model = YOLO("yolov10n.pt")
# Define the base directory for YOLO training runs
base_dir = "runs/detect"
# Find the latest run directory by looking for the most recently modified directory
run_dirs = sorted(glob.glob(os.path.join(base_dir, "train*")), key=os.path.getmtime)

# Set training args based on what you need https://docs.ultralytics.com/modes/train/#train-settings
epochs = 25
imgsz = 640
batch = 8 * torch.cuda.device_count() # Must be a multiple of GPU numbers !!!
# If you get out-of-memory errors, try reducing the batch size.
device = [i for i in range(torch.cuda.device_count())] # Use all available gpus
data = "coco8.yaml" # Change to your desired yaml

if run_dirs:
    latest_run_dir = run_dirs[-1]
    # Check if a checkpoint exists in the latest run directory
    last_checkpoint_path = os.path.join(latest_run_dir, "weights", "last.pt")
    print(f">>> FROM SCRIPT >>> Latest run directory: {latest_run_dir}")
    print(f">>> FROM SCRIPT >>> Last checkpoint: {last_checkpoint_path}")

    # Last.pt exists
    if os.path.exists(last_checkpoint_path):
        model2 = YOLO(last_checkpoint_path)
        start_epoch = model2.ckpt['epoch']
        # Finished training all epochs, start a new one
        if (start_epoch < 0):
            # Train model again. Use different train parameters if needed.
            print(f">>> FROM SCRIPT >>> Last.pt completed all epochs, starting new training...")
            results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)
        else:
            # Resume interrupted training
            print(f">>> FROM SCRIPT >>> Interrupted, resuming training...")
            results = model2.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=True)
    
    else:
        print(">>> FROM SCRIPT >>> No last.pt found. Starting new training...")
        results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)

else:
    print(">>> FROM SCRIPT >>> No previous run directories found. Starting new training...")
    results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)