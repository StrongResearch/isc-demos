## Training YOLOv10 on a Single Machine with Multiple GPUs
**Note that this is a small demo run using the COCO8 dataset. If you want to train YOLOv10 on your own dataset, scroll down to the Custom Dataset Training section once you have completed Setup.**

## Setup
1. Navigate to home directory, create a virtual environment and activate it.
```
cd ~
python3 -m virtualenv .yolov10-train
source .yolov10-train/bin/activate
```
2. Install python package requirements
```
cd ~/isc-demos/yolov10-training
pip install -r requirements.txt
```
3. Make sure you can use GPUs for training by checking in python. If you can, the output here should be 'True'
```
python3 -c "import torch; print(torch.cuda.is_available())"
```
4. Update the `yolov10-train.isc` file with your own project ID. If you wish to change the GPU numbers, note that you can only go up to maximum capacity for a single machine. You can change the experiment name too if you want.

5. For this demo, we'll be training YOLOv10n on the COCO8 dataset. You should not need to download them, as Ultralytics handles these automatically.

## Running the script
6. Start an experiment with the following command
```
isc train yolov10-train.isc
```
7. YOLO should output a line like:
```
Logging results to runs/detect/train2
```
After the training is complete, check your results at that specified directory. You should be able to see images of the confusion matrix and overall results graphs.

# Custom Dataset Training
1. Organize your dataset like this. If you do not have a test split for your data, it is OK. Just make sure you have the validation split.
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
2.  Next, create a YAML file for your dataset that matches this format: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/?h=custom+da#21-create-datasetyaml. Ensure that the labels for your images follows the format outlined in step 2.2 of the above link.

3. Make sure that YOLO knows where your dataset is located. You can either update the `datasets_dir` field in `/root/.config/Ultralytics/settings.json` or move your dataset into the directory specified by `datasets_dir`

4. If you want to change the YOLO model, go to the `yolov10-train.py` script and change this line to your desired model (note that yolov8 and yolov9 should also work).
```py
model = YOLO("yolov10n.pt")
```
5. Change this line to the absolute/relative path of your dataset's YAML file. I recommend using absolute path.
```py
data = "coco8.yaml" # Change to your desired yaml
```
6. In the python script, you are free to change the number of epochs, batch size and any other training parameters. You can see the full list of arguments here: https://docs.ultralytics.com/modes/train/#train-settings

7. For lines 33-36, if you want, you can change the logic to use the resulting `best.pt` model from the previous run in a new training run like so:
```py
    if (start_epoch < 0):
        # Feed best.pt from last training run into this run
        best_result_path = os.path.join(latest_run_dir, "weights", "best.pt")
        model3 = YOLO(best_result_path)
        print(f">>> FROM SCRIPT >>> Last.pt completed all epochs, starting new training...")
        results = model3.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)
```
8. For bigger datasets, you should change `yolov10-train.isc`'s training mode to "interruptible". `yolov10-train.py` should take care of resuming interrupted training.
9. Start the training with
```
isc train yolov10-train.isc
```
10. Once training is done check your training directory for results. The resulting model you want is located at `weights/best.pt`.