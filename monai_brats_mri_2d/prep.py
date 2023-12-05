# Ensure required monai version is installed
# pip install monai==1.2.0

# Download the toy dataset from MONAI
print("Downloadning BraTS2016/17")
from monai.apps import DecathlonDataset
from monai import transforms
from generative.losses.perceptual import PerceptualLoss
import h5py
from tqdm import tqdm

# Cache pre-trained weights for perceptual loss
perceptual_loss = PerceptualLoss(
    spatial_dims=2, network_type="resnet50", pretrained=True, #ImageNet pretrained weights used
)

data_path = "/mnt/.node1/Open-Datsets/MONAI"
_ = DecathlonDataset(root_dir=data_path, task="Task01_BrainTumour", section="training", download=True)

# # Download the bigger dataset from Synapse
# print("Downloadning BraTS2023")
# import synapseclient
# syn = synapseclient.Synapse()
# syn.login('adam_peaston','AXXXXXXXXX2')
# syn51514132 = syn.get(entity='syn51514132', downloadFile=True, downloadLocation="/mnt/Datasets/strongcompute_adam/MONAI", ifcollision="overwrite.local")
# filepath = syn51514132.path
# print(f"BraTS2023-GLI downloaded to {filepath}")

# Simulating dataset and dataloader from original training script
channel = 0  # 0 = "Flair" channel
assert channel in [0, 1, 2, 3], "Choose a valid channel"
preprocessing_transform = transforms.Compose([
        transforms.LoadImaged(keys="image", image_only=False), # image_only current default will change soon, so including explicitly
        transforms.EnsureChannelFirstd(keys="image"),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.AddChanneld(keys="image"),
        transforms.EnsureTyped(keys="image"),
        transforms.Orientationd(keys="image", axcodes="RAS"),
        transforms.CenterSpatialCropd(keys="image", roi_size=(240, 240, 100)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100, b_min=0, b_max=1),
])
        
crop_transform = transforms.Compose([
        transforms.DivisiblePadd(keys="image", k=[4,4,1]),
        # transforms.RandSpatialCropSamplesd(keys="image", random_size=False, roi_size=(240, 240, 1), num_samples=26), # Each of the 100 slices will be randomly sampled.
        # transforms.SqueezeDimd(keys="image", dim=3),
        # transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=0),
        # transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=1),
])

preprocessing = transforms.Compose([preprocessing_transform, crop_transform])

train_ds = DecathlonDataset(
    root_dir=data_path, task="Task01_BrainTumour", section="training", cache_rate=0.0,
    num_workers=8, download=False, seed=0, transform=preprocessing,
)
val_ds = DecathlonDataset(
    root_dir=data_path, task="Task01_BrainTumour", section="validation", cache_rate=0.0,
    num_workers=8, download=False, seed=0, transform=preprocessing,
)

# Loop over the dataset and store in HDF.
hf = h5py.File('dataset.h5', 'a')

for i,train_data in enumerate(tqdm(train_ds, desc='Compiling training data')):
    X_train = train_data["image"][0,:,:,:].permute(2, 1, 0)
    if i == 0:
        hf.create_dataset("train", data=X_train, maxshape=(None, 240, 240))
    else:
        hf['train'].resize((hf['train'].shape[0] + X_train.shape[0]), axis=0)
        hf['train'][-X_train.shape[0]:] = X_train
    
for i,val_data in enumerate(tqdm(val_ds, desc='Compiling validation data')):
    X_val = val_data["image"][0,:,:,:].permute(2, 1, 0)
    if i == 0:
        hf.create_dataset("val", data=X_val, maxshape=(None, 240, 240))
    else:
        hf['val'].resize((hf['val'].shape[0] + X_val.shape[0]), axis=0)
        hf['val'][-X_val.shape[0]:] = X_val

print(hf["train"].shape)
print(hf["val"].shape)

hf.close()