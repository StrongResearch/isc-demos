# ISC Demos
Welcome to the Strong Compute ISC Demos repo, here you will find all the instructions you need to get set up to train 
Pytorch models on the Strong Compute ISC.
 - [1. Getting Started](#getting-started)
   - [1.1. Setting up the VPN](#setting-up-vpn)
     - [1.1.1. For MacOS and Windows](#for-mac-windows)
     - [1.1.2. For Linux](#for-linux)
   - [1.2. Creating your ISC User and Organisation credentials](#creating-your-isc-user-credentials)
 - [2. Interruptible Experiments](#interruptible-experiments)
   - [2.1. Rapid Cycling and Burst To Cloud](#rapid-cycling-burst)
   - [2.2. Hello World with Fashion MNIST](#hello-world-with-fashion-mnist)
   - [2.3. More examples](#more-examples)
 - [3. Transferring your dataset](#data-transfer)


## 1. Getting started <a name="getting-started"></a>

### 1.1. Setting up the VPN <a name="setting-up-vpn"></a>
Before connecting to the Strong Compute ISC, you must have recieved login credentials from Strong Compute by email. 
Please reach out to us if you have not recieved this email.

#### 1.1.1. For MacOS and Windows <a name="for-mac-windows"></a>

1. You will need to download and install WireGuard from https://www.wireguard.com/install/.
2. Once you have WireGuard installed, visit the Strong Compute FireZone portal at the website advised in the email (above) and 
    login with the credentials you receieved by email.
3. In the FireZone portal, click "Add Device". You can give the configuration any name you like, if you do not provide a 
    name for the configuration, FireZone will automatically generate one for you.
4. Click on "Generate Configuration" and then "Download WireGuard Configuration" to download the tunnel config file. 
    Save the tunnel config file somewhere suitable on your machine.
5. Within the WireGuard application, select "Import Tunnel(s) from File" and select the tunnel config file from where 
    you saved it.
6. Ensure the VPN tunnel is enabled when accessing the Strong Compute ISC. When the VPN is correctly enabled you should 
    be able to open a terminal, run `ping 192.168.127.70` and recieve `64 bytes from 192.168.127.70`.

#### 1.1.2. For Linux <a name="for-linux"></a>

1. You will need to download and install WireGuard from https://www.wireguard.com/install/.
2. Once you have WireGuard installed, visit the Strong Compute FireZone portal at the website advised in the email (above) and 
    login with the credentials you receieved by email.
3. In the FireZone portal, click "Add Device". You can give the configuration any name you like, if you do not provide a 
    name for the configuration, FireZone will automatically generate one for you.
4. Click on "Generate Configuration" and then "Download WireGuard Configuration" to download the tunnel config file. 
    Save the tunnel config file somewhere suitable on your machine.
5. Within your terminal run the following commands (you will need sudo-level access):
 - Ensure the package manager is up-to-date with `sudo apt update`
 - Install WireGuard with `sudo apt install -y wireguard`
 - Open the WireGuard config file with `sudo nano /etc/wireguard/wg0.conf`
6. In the WireGuard config file, delete or comment out the line starting with "DNS" (you can comment out by adding a `#` 
    to the start of the line).
7. Open the tunnel config file you downloaded from FireZone portal, copy and paste the contents of the tunnel config 
    file into the WireGuard config file. Your WireGuard config file should look as follows.

```
# DNS = 1.1.1.1,10.10.10.1

[Interface]
PrivateKey = <private-key>
Address = <ip address>
MTU = 1280

[Peer]
PresharedKey = <preshared-key>
PublicKey = <public-key>
AllowedIPs = <ip address>
Endpoint = <endpoint>
PersistentKeepalive = 15
```

8. Save the updated WireGuard config file and close nano.
9. Ensure the VPN tunnel is enabled when accessing the Strong Compute ISC. You can enable the VPN with 
    `sudo wg-quick up wg0` and disable with `sudo wg-quick down wg0`. When the VPN is correctly enabled you should be 
    able to open a terminal, run `ping 192.168.127.70` and recieve `64 bytes from 192.168.127.70`.

### 1.2. Creating your ISC User and Organisation credentials  <a name="creating-your-isc-user-credentials"></a>
Now that you have your VPN correctly installed, configured, and enabled, we'll get you set up with a login to the ISC 
associated with your User and Organisation.

**If you are the first person in your Organisation to register:**

1. Visit the Strong Compute **Control Plane** at https://cp.strongcompute.ai/, click on **"Register"**, register with your email 
    and choose a suitable password.
2. Click on the menu at the top right of the page (labelled with your email and Organisation name) and click on **"Organisations"**. 
    On the "Your Organisations" page, click on the tile for your organisation. When you first register, your Organisation will be
    named **"\<your email\>'s Personal Organisation"**.
3. From the **"Organisation Settings"** page you can update the name of your
    Organisation, view current members, and *create* invitations for other members to join. **Note that Control Plane does not send
    emails to invitees.** If you invite a member to join your Organisation, you will need to notify them that they have been invited
    and which email address their invitation is registered with.

**If you are responding to an invitation to join an Organisation on Control Plane, continue from here:**
   
1. Visit the Strong Compute **Control Plane** at https://cp.strongcompute.ai/, click on **"Register"**, register with your email 
    and choose a suitable password.
2. Click on the menu at the top right of the page (labelled with your email and Organisation name) and click on **"Organisations"**.
3. Your invitation will be visible on the Organisations page with options to Accept or Reject.

**Instructions for all users continue from here:***
   
1. On Control Plane click on the menu at the top right of the page and click on **"Settings"**.
2. At the bottom of the page, click on **"NEW SSH KEY"**.
3. You will need a cryptographic key pair which you can generate by opening a terminal and running `ssh-keygen`. When prompted,
    provide a filename in which to save the key. You can also optionally enter a passphrase (or just press enter twice). This will 
    generate two files containing your public and private keys. Your public key will be saved in the file ending in `.pub`.
    Alternatively you can use any 
4. Open the file containing your public key. If you generated your keypair with the instruction above, the public key file contents
    should start with `ssh-rsa` and end with `.local`. Copy the entire contents of this file and paste into the input field on
    Control Plane beneath **"Public key contents"**, then click **"SUBMIT NEW SSH PUBLIC KEY"**.
5. Return to the Settings page on Control Plane and click on **"NEW API KEY"**. Optionally name your API key. If you already have
    multiple Organisations established, select the Organisation to associate this API key with. Click on **"GENERATE NEW API ACCESS
    TOKEN"**. You will be presented with the API Access Token associated with your API key. Save a copy of this API Access Token. Be
    careful to save the entire API Access Token, note that some characters will be obscured by the fixed width of the display window.
    For security purposes you will only be shown this API Access Token once and will not be able to access it again, though you can
    always create a new API Key. Note you will need a separate API Key set up for each Organisation you wish to login to.
6. Click on **"Back to Settings"**. You should see the new API Key that you just created, and an associated SSH Username. 
    You will use the command shown under **"SSH Username"** to connect to the ISC via SSH.
7. <a name="org-id"></a> Open a terminal and enter the entire the SSH Username command. The command should start with `ssh` and end 
    with `@<ip-address>`. You should be greeted by the Strong Compute logo and ISC welcome message below. This indicates that you have 
    successfully logged into your home directory on the ISC. Your home directory on the ISC is a subdirectory within your Organisation
    directory. Running `pwd` you will see the full path to your home directory following the pattern `/mnt/Client/<OrgID>/<UserID>`.

```bash
                    ;≥░░░≥≥-
             ╓ ]▒╠╬╦  ░░░Γ ,φ╬╬▒  ,
      ,╓ )▒╬╬╬╦ ╚╬╬╬╬╦   ,φ╬╬╬╬╜ φ╬╬╬▒ ┌,
    ╬╬╬╬  ╬╬╬╬╬╦ ╙╬╬╩  ╔╖ ╙╩╬╬╙ φ╬╬╬╬╩ ╚╬╬╬▒
    ╠╬╬╬▒ ╚╬╬╬╬╬▒   ╓φ╬╬╬╠╦,   ╬╬╬╬╬╬  ╬╬╬╬╬
    ╘╬╬╬╬ε ╠╬╩╙ ,╔ç ╚╬╬╬╬╬╬╜ ╔╖  ╙╩╬╜ ╠╬╬╬╬╩
     ╬╬╬╩╙  ,╔φ╬╬╬╬▒  ╚╬╬╩ ,╠╬╬╬╬╦╖   ╙╩╬╬╬
       ,╔φ▒  ╬╬╬╬╬╬╬╬╦   ,φ╬╬╬╬╬╬╬╩ ╔▒╦╖,
      ╠╬╬╬╬╬ç ╠╬╬╬╬╬╬╜ ╓╖ ╙╬╬╬╬╬╬╙ φ╬╬╬╬╬╬
       ╬╬╬╬╬╬╦ ╙╬╩╜ ,φ╬╬╬╬╗  ╙╩╬  ╠╬╬╬╬╬╬
        ╠╬╬╬╬╩╜  .φ╬╬╬╬╬╬╬╬╬╬╗   ╙╩╬╬╬╬╬
         ╙  ,╓#▒▒  ╚╬╬╬╬╬╬╬╬╜ ╓╬▒╗╖   ╙
           ╬╬╬╬╬╬╬▒  ╙╠╬╬╬╜ ╓╬╬╬╬╬╬╬╬
            ╙╬╬╬╬╬╬╬▒╖   ,φ╬╬╬╬╬╬╬╬╩
              ╙╬╬╬╬╝╜ ,╔╖  ╙╬╬╬╬╬╩
                ╙  ╓#▓╬╬╬╬▒╗╖  ╙
                   ╙╣╬╬╬╬╬╬╬╜
                      ╙╝╝╙
 
=================================================
                ISC v0.5.0-alpha
=================================================

Version 0.5.0-alpha of the ISC is now live!

## Changelog
...
## Cycling
...
## Checkpointing
...
## Cycling utilities
...
## Example training scripts
...
```

8. Run `isc login` and enter the API Key you saved previously at step 6. This will create your `credentials.isc` file which 
    is used to authenticate you when you launch experiments on the ISC, and should be saved in the root of your home directory.
9. Run `isc ping` and you should receieve `Success: {'data': 'pong'}` in response to indicate that your credentials file has been 
    created correctly.
10. Create and activate a virtual environment by running the following commands.

```bash
python3 -m virtualenv ~/.venv
source ~/.venv/bin/activate
```
11. You will need to ensure that you have activated your virtual environment when installing 
    dependencies for your experiments, and that the path to your virtual environment directory is included correctly in your [ISC
    Config file(s)](#isc-config).
12. Only Pytorch models are currently supported on the ISC and distributed training is coordinated using torchrun, so install pytorch 
    by running `pip install torch==2.0.1`.

Congratulations, you are all set to start running experiments on the ISC. Follow the next steps in this guide to 
configure and launch your first "hello world" experiment, and learn about necessary steps to make sure your experiment 
is "interruptible" (including what this means).

## 2. Interruptible experiments <a name="interruptible-experiments"></a>

We will now explore and run some code to launch an experiment on the ISC. This example will demonstrate the principle and 
application of interruptibility, which will be important to consider when developing your experiments to run successfully 
on the ISC.

### 2.1. Rapid Cycling and Burst To Cloud <a name="rapid-cycling-burst"></a>

The ISC is comprised of a **Rapid Cycling** stage and a **Burst To Cloud ("Burst")** stage. Experiments launched on the ISC are 
run first in **Rapid Cycling** as a validation step before being **Burst** to the cloud. This provides Users with near-immediate 
feedback on the viability of their code *before* their experiment is launched in a dedicated cloud cluster and incurs costs.

**Action:** Run `isc experiments` to see all of your historic experiments in the **Experiments Table**. When you first register, 
this table will be empty. Each time you launch an experiment on the ISC, a record of that experiment will appear in this 
table. Only *your* experiments will be visible to you in this table. There may be other experiments scheduled to run on 
the ISC at the same time that will not be visible to you.

```bash
                               ISC Experiments                                                                                                                          
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ ID     ┃ Name     ┃ NNodes    ┃ Output Path                                 ┃ Status        ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│        │          │           │                                             │               │
└────────┴──────────┴───────────┴─────────────────────────────────────────────┴───────────────┘
```

Experiments share time on the **Rapid Cycling** cluster by means of a queue which is cycled at fixed time intervals. The 
queue is comprised of two sub-queues, indicated by status `enqueued` and `paused` respectively. All `enqueued` 
experiments are cycled before any `paused` experiment is cycled.

When a new experiment is launched onto the Rapid Cycling cluster, it joins the `enqueued` queue. While the experiment is 
running, its status will be recorded as `running`, and it will be permitted to run for a fixed period of **90 seconds**. 
If the experiment **fails** during its running cycle (for example if there was an error in the experiment code) then the 
experiment status will be recorded as `failed` and the experiment will be removed from the queue. Otherwise, at the end 
of this period, the experiment will be **paused** to allow other experiments to cycle, and it will join the `paused` queue. 
Experiments from the `paused` queue are cycled when there are no experiments waiting in the `enqueued` queue. 

Once an experiment has **completed 5 cycles** it will be scheduled for **Burst**. This means a dedicated cluster will be
created for the experiment in the cloud, and the experiment will be launched on this dedicated cluster to train. In case of 
cloud interruption (for example due to blackout or hardware failure), another dedicated cluster will be created and the 
experiment resumed on the new dedicated cluster.

The ability to abruptly interrupt experiments or **"interruptibility"** is crucial for the purpose of **Rapid Cycling**, pausing 
and resuming. The main approach to achieve interruptibility is **robust and frequent checkpointing** which we will 
demonstrate with the example project that follows.

### 2.2. Hello World with Fashion MNIST <a name="hello-world-with-fashion-mnist"></a>

To follow this demonstration, first ensure you have activated your virtual environment, cloned this repo in your home 
directory on the ISC, and installed the necessary requirements with the following commands. You will also need to clone the 
`cycling_utils` repo (https://github.com/StrongResearch/cycling_utils) and install it as a package in editable mode.

```bash
cd ~
source ~/.venv/bin/activate
git clone https://github.com/StrongResearch/isc-demos.git
git clone https://github.com/StrongResearch/cycling_utils.git
pip install -e cycling_utils
```

The `cycling_utils` package contains helpful functions and classes for achieving interruptibility in distributed training. 
Installing `cycling_utils` in editable mode will allow you to extend this package at any time with your own modules as needed 
without having to reinstall the package. Next navigate to the `isc-demos` repo directory and inspect the contents of the 
`fashion_mnist` subdirectory.

```bash
cd ~/isc-demos/fashion_mnist
ls
```

The `isc-demos/fashion_mnist` subdirectory contains the following files of interest.
1. `requirements.txt` includes dependencies necessary for this specific demo. Install these by running `pip install -r 
    requirements.txt`.
2. `prep_data.py` includes commands for downloading the required dataset (Fashion MNIST). Run this with `python -m prep_data` 
    to download the dataset to the `fashion_mnist` directory, from which it is available to the Rapid Cycling cluster. This will 
    mean that this data is ready to go when the experiment is launched and no cycling time is wasted waiting for the data to 
    download. Preparing your data ahead of time is an essential requirement for running experiments on the ISC and we will cover 
    [**how to transfer your private dataset to our cloud storage**](#data-transfer) for training in a later section.
3. `model.py` includes a description of the neural network model that we will train.
4. `train.py` describes configuration for distributed training, initialisation, and distributed training loops. Take a 
    few minutes to read and understand the contents of this file, there are lots of notes to explain what's happening. 
    Reach out with any questions. Note that `train.py` provides for command line arguments to be passed, we will see how 
    when looking at the next file.
5. `fashion_mnist.isc` is the **ISC Config** <a name="isc-config"></a> file necessary for launching this experiment on the isc. 
   The key information included in the ISC Config file is as follows.
    - `experiment_name`: This name will appear in the **Experiments Table**. Use this name to uniquely identify this 
        experiment from your other experiments at a glance, for example by encoding hyper-parameters that you are testing.
    - `gpu_type`: The type of GPU that you are requesting for your experiment. At this time, the only supported 
        `gpu_type` is **`24GB VRAM GPU`**, so leave this unchanged.
    - `nnodes`: This is the number of nodes that you are requesting. Each node will have a number of GPUs. The **Rapid 
        Cycling** cluster nodes each have **6 GPUs**, and there are currently a **maximum of 12 nodes available**, so be 
        sure to request between 1 and 12 nodes.
    - `venv_path`: This is the path to your virtual environment which we set up above. If you followed the instructions 
        above for **"Creating your ISC User and Organisation credentials"** then you should leave this unchanged.
    - `output_path`: This is the directory where the outputs of your experiment will be saved, including reports from 
        each node utilised by your experiment. It can be helpful to consider how you would like to structure your 
        output(s) directory to keep your experiment results organised (i.e. subdirectories for experiment groups).
    - `command`: This is the launch command passed to **torchrun** to commence your experiment. Note the call to `train.py` 
        and the command line arguments passed. Refer to the notes within `train.py` to understand how these arguments 
        affect training and checkpointing.

After you have run `prep_data.py` (above) to pre-download the Fashion MNIST dataset, you can launch this experiment to 
train on the ISC using the following command.

```bash
isc train fashion_mnist.isc
```

You should recieve the response `Success: Experiment created`. Running `isc experiments` you should be able to see your 
experiment in the experiments table with the status `enqueued`. Other details about your experiment displayed include 
the following.
 - `ID`: The UUID assigned to uniquely idenfity your experiment in our database.
 - `Name`: The experiment name you specified in the `.isc` file, `fashion_mnist` in this case.
 - `NNodes`: The number of nodes requested for your experiment.
 - `Output Path`: The path to your experiment output directory. It may be necessary to zoom out to see the full path.

 Refresh the Experiments Table periodically by running `isc experiments`. When you see your experiment transition from 
 `enqueued` to `running`, navigate to the experiment output directory. You will find a number of `rank_X.txt` files, 
 corresponding to each of the `NNodes` nodes requested for your experiment. You can thus verify that all requested nodes 
 were initialised. Each `rank_X.txt` file should contain at least the following. By default, only the `rank_0.txt` file 
 should contain anything more.

 ```bash
 WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being \
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
 ```

 **Action:** Open the **`rank_0.txt`** file and inspect the contents. Compare the outputs in this file with the reporting 
 points within `train.py`. If there are errors in your code, the `rank_X.txt` files will be where to look to understand 
 what has gone wrong and how to fix it.

 You will also find a `checkpoint.isc` file saved in the experiment output directory. This is the checkpoint file that is 
 regularly updated with the information necessary to resume your experiment after it is paused.

 Lastly you will find a subdirectory called `tb` which contains tensorboard event logs. Refer to the `train.py` file to 
 understand where and how these event logs are created.

 Returning to the command line, you can launch a tensorboard instance to track the training and performance metrics of 
 the experiment with the following command.

 ```bash
 tensorboard --logdir <Output Path from ISC Experiments table>
 ```

Tensorboard will attempt to launch on a default port (typically 6006). If this port is unavailable, you can manually set an alternative port of your chosing with the flag `--port <port>`. You can then view the tensorboard at `http://localhost:<port>/`. Tensorboard recursively searches for tensorboard logs in the directory passed after the `--logdir` flag, so it will discover the event logs in the `/tb` subdirectory.

![fashion_mnist_tensorboard](https://github.com/StrongResearch/isc-demos/blob/main/fashion_mnist/fashionmnist_tensorboard.png?raw=true)

Continue to track the progress of your experiment while it cycles by checking in on the `rank_0.txt` file and the 
tensorboard.

Congratulations, you have successfully launched your first experiment on the ISC!

### 2.3. More examples <a name="more-examples"></a>

The following examples further demonstrate how to implement interruptibility in distributed training scripts using 
checkpointing, atomic saving, and stateful samplers.

These examples are being actively developed to achieve [1] interruptibility in distributed training, [2] verified 
completion of a full training run, and [3] achievement of benchmark performance published by others (where applicable). 
Each example published below is annotated with its degree of completion. Examples annotated with [0] are "coming soon".

#### Hello World

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| Fashion MNIST | Image classification | CNN | [3] | [isc-demos/fashion_mnist](fashion_mnist) |
| CIFAR100 | Image classification | ResNet50 | [2] | [isc-demos/cifar100-resnet50](cifar100-resnet50) |
| Distributed Model Parallel | TBC | TBC | [0] | |

#### pytorch-image-models (timm)

(from https://github.com/huggingface/pytorch-image-models)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| resnet50 | Image classification | ResNet50 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| resnet152 | Image classification | ResNet152 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnet_b0 | Image classification | EfficientNet B0 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnet_b7 | Image classification | EfficientNet B7 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnetv2_s | Image classification | EfficientNetV2 S | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnetv2_xl | Image classification | EfficientNetV2 XL | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| vit_base_patch16_224 | Image classification | VIT Base Patch16 224 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| vit_large_patch16_224 | Image classification | VIT Large Patch16 224 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |

#### Torchvision segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| fcn_resnet101 | Image segmentation | ResNet101 | [2] | [isc-demos/tv-segmentation](tv-segmentation) |
| deeplabv3_mobilenet_v3_large | Image segmentation | MobileNetV3 Large | [2] | [isc-demos/tv-segmentation](tv-segmentation) |

#### Torchvision detection

(from https://github.com/pytorch/vision/tree/main/references/detection)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| maskrcnn_resnet101_fpn | Object detection | Mask RCNN (ResNet101 FPN) | [2] | [isc-demos/tv-detection](tv-detection) |
| retinanet_resnet101_fpn | Object detection | RetinaNet (ResNet101 FPN) | [2] | [isc-demos/tv-detection](tv-detection) |

#### Detectron2

(from https://github.com/facebookresearch/detectron2)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| detectron2 | TBC | Detectron2 | [2] | [isc-demos/detectron2](detectron2) |
| detectron2_densepose | TBC | Detectron2 | [2] | [isc-demos/detectron2/projects/densepose](detectron2/projects/densepose) |

#### Large Language Models (LLM)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| Llama2 | LoRA | Llama2 | [0] | [isc-demos/llama2](llama2) |
| Mistral | TBC | Mistral | [0] | [isc-demos/mistral](mistral) |

## 3. Transferring your dataset <a name="data-transfer"></a>
The process for transferring large datasets to the ISC for training includes two main steps:
1. Download your dataset to the **Download Server**.
2. Transfer your dataset to your Organisation directory on one of our **Data Nodes**.

**Note:** We will need to advise you on which **Data Node** to transfer your data to. Please contact us to discuss your dataset 
and be assigned a data node. It is important to note that all Users within your Organisation will have access to datasets saved 
in your Organisation directory on the Download Server and the Data Node.

Use the following command to SSH into the Download Server.
```bash
ssh username@192.168.127.100
```

Download your data to your Organisation directory on the Download Server. You can obtain your [OrgID from the full path 
to your home directory](#org-id).
```bash
/Downloads/<OrgID>/<dataset-name>
```

Move your data to your Organisation directory on your assigned Data Node.
```bash
/mnt/.node<assigned-node>/<OrgID>/<dataset-name>
```

Make note of the path to your dataset and ensure that your training scripts correctly reference this path when loading data.
