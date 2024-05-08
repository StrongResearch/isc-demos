# ISC Demos
Welcome to the Strong Compute Instant Super Computer (ISC) Demos repo, here you will find all the instructions you need to get set up to train 
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
 - [3. Data Parallel Scaling](#data-parallel-scaling)
 - [4. Uploading your dataset](#data-transfer)


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
   
1. On Control Plane click on the menu at the top right of the page and click on **"User Credentials"**.
2. At the bottom of the page, click on **"ADD SSH KEY"**.
3. You will need a cryptographic key pair which you can generate by opening a terminal and running `ssh-keygen`. When prompted,
    provide a filename in which to save the key. You can also optionally enter a passphrase (or just press enter twice). This will 
    generate two files containing your public and private keys. Your public key will be saved in the file ending in `.pub`.
4. Open the file containing your public key. If you generated your keypair with the instruction above, the public key file should 
    have a file extension of `.pub`. Copy the entire contents of this file and paste into the input field on Control Plane beneath 
    **"Public key contents"**, then click **"SUBMIT NEW SSH PUBLIC KEY"**.
5. Return to the Settings page on Control Plane and click on **GENERATE** under the heading **Containers**. Strong Compute User 
   environments are accessed via Docker containers, and this will generate you a new base image. Once your container has been created, 
   you will then see the button update to allow you to **START** your container.
6. You will also see displayed a new **ssh command** which will include a specified port number and the login **root@192.168.127.70**
   which you can use to access your User environment. You may need to extend this ssh command to specify the private key to use as follows.
   If you submitted your public key saved at `~/.ssh/id_rsa.pub` then you should be able to ignore this requirement.

```bash
# If not submitted ~/.ssh/id_rsa.pub
ssh -p <port> root@192.168.127.70 -i <path/to/your/private_key>

# If submitted ~/.ssh/id_rsa.pub
ssh -p <port> root@192.168.127.70
```
7. Note that your container **must be started** in order for you to connect to it via ssh, and a **new port number** will be assigned each
   time your container is stopped and started.
8. Please ignore the **Access Tokens** section of the User Credentials page. This is deprecated and will be removed shortly.
9. <a name="isc-project-id"></a> From the main page tabs, click on **"Projects"**. All experiments launched on the ISC must 
    be associated with one and only one **ISC Project** which is used for usage tracking and cost control. Click on 
    **"NEW PROJECT"** and give your new **ISC Project** a name. You will also need the help of your Organisation Owner or Admins 
    to ensure your Organisation has sufficient credits and that cost controls have been set to permit experiments to be 
    launched under your new **ISC Project**.
10. <a name="org-id"></a> Open a terminal and enter the ssh command obtained above. Congratulations, you are all set to start running 
    experiments on the ISC. Follow the next steps in this guide to configure and launch your first "hello world" experiment, and learn
    about necessary steps to make sure your experiment is "interruptible" (including what this means).

**Note:** Your environment is running in a **Docker container**. Your home directory is **read/write** mounted at `/root`. A shared directory accessible 
to all members of your Organisation is **read/write** mounted at `/shared`. You can install any software you require in your container
which will be committed and pushed to our docker registry when you **STOP** your container and when you launch an experiment.

We recommend saving all working files (including virtual environments) to `/root` in order to **minimise the size** of your container, thus 
minimizing the start time for your container and the time to launch your experiments on the ISC.

## 2. Interruptible experiments <a name="interruptible-experiments"></a>

We will now explore and run some code to launch an experiment on the ISC. This example will demonstrate the principle and 
application of interruptibility, which will be important to consider when developing your experiments to run successfully 
on the ISC.

### 2.1. Rapid Cycling and Burst To Cloud <a name="rapid-cycling-burst"></a>

The ISC is comprised of a **Rapid Cycling** stage and a **Burst To Cloud ("Burst")** stage. Experiments launched on the ISC are 
run first in **Rapid Cycling** as a validation step before being **Burst** to the cloud. This provides Users with near-immediate 
feedback on the viability of their code *before* their experiment is launched in a dedicated cloud cluster and incurs costs.

**Action:** Run the following to see all of your historic experiments in the **Experiments Table**. 
```bash
isc experiments
```
When you first register, this table will be empty. Each time you launch an experiment on the ISC, a record of that experiment will 
appear in this table. Only *your* experiments will be visible to you in this table. There may be other experiments scheduled to run 
on the ISC at the same time that will not be visible to you.

```bash
                               ISC Experiments                                                                                                                          
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ ID     ┃ Name     ┃ NNodes    ┃ Output Path                                 ┃ Status        ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│        │          │           │                                             │               │
└────────┴──────────┴───────────┴─────────────────────────────────────────────┴───────────────┘
```

Experiments share time on the **Rapid Cycling** cluster by means of a queue which is cycled at fixed time intervals. The 
queue is comprised of two sub-queues, a **rapid cycling** queue and an **interruptible** queue.

When a new experiment is launched onto the Rapid Cycling cluster, it joins the **rapid cycling queue**. When it is this
experiment's turn to run, its status will be recorded as `running` in the **Experiments Table**. It will run for **90 seconds**
and then will be paused and placed at the back of the **rapid cycling queue**, allowing the next experiment to run. Experiments 
in the **rapid cycling queue** cycle in this fashion until they have been run 5 times, and then are moved into the 
**interruptible queue**. Experiments in the **interruptible queue** are cycled when there are no experiments waiting in the 
**rapid cycling queue**, and are then considered ready to be scheduled for **Burst**.

If the experiment **fails** during any running cycle (for example if there was an error in the experiment code) then the 
experiment status will be recorded as `failed` in the **Experiments Table** and the experiment will be dropped from the 
queue it was in.

When an experiment is scheduled for **Burst**, a dedicated cluster will be created for the experiment in the cloud, and 
the experiment will be launched on this dedicated cluster to train. In case of cloud interruption (for example due to blackout 
or hardware failure), another dedicated cluster will be created and the experiment resumed on the new dedicated cluster.

The ability to abruptly interrupt experiments or **"interruptibility"** is crucial for the purpose of **Rapid Cycling**, 
pausing, and resuming. The main approach to achieve interruptibility is **robust and frequent checkpointing** which we will 
demonstrate with the example project that follows.

### 2.2. Hello World with Fashion MNIST <a name="hello-world-with-fashion-mnist"></a>

To follow this demonstration, first ensure you have cloned this repository in your home directory on the ISC.

The first step when commencing a new project on the ISC is to create and activate a virtual environment as follows.

```bash
cd ~
python3 -m virtualenv ~/.fashion
source ~/.fashion/bin/activate
```

Next we will clone this repo to access the example source code and navigate to the fashion_mnist subdirectory.
```bash
git clone https://github.com/StrongResearch/isc-demos.git
```

The `isc-demos/fashion_mnist` subdirectory contains the following files of interest.
1. `requirements.txt` includes dependencies necessary for this specific demo, including the version of `pytorch` necessary 
for our project. Install these by running the following. Note, only Pytorch models are currently supported on the ISC and 
distributed training is coordinated using torchrun. 
```bash
cd ~/isc-demos/fashion_mnist
pip install -r requirements.txt
```
Note the `cycling_utils` package among the installed dependencies. The `cycling_utils` package contains helpful functions and 
classes for achieving interruptibility in distributed training. Installing `cycling_utils` in editable mode will allow you to 
extend this package at any time with your own modules as needed without having to reinstall the package. Next navigate to the 
`isc-demos` repo directory and inspect the contents of the `fashion_mnist` subdirectory.

2. `prep_data.py` includes commands for downloading the required dataset (Fashion MNIST). Run this with the following command 
    to download the dataset to the `fashion_mnist` directory, from which it is available to the Rapid Cycling cluster. This will 
    mean that this data is ready to go when the experiment is launched and no cycling time is wasted waiting for the data to 
    download. Preparing your data ahead of time is an essential requirement for running experiments on the ISC and we will cover 
    [**how to transfer your private dataset to our cloud storage**](#data-transfer) for training in a later section.
```bash
cd ~/isc-demos/fashion_mnist
python -m prep_data
```
3. `model.py` includes a description of the neural network model that we will train.
4. `train.py` describes configuration for distributed training, initialisation, and distributed training loops. Take a 
    few minutes to read and understand the contents of this file, there are lots of notes to explain what's happening. 
    Reach out with any questions. Note that `train.py` provides for command line arguments to be passed, we will see how 
    when looking at the next file.
5. `fashion_mnist.isc` is the **ISC Config** <a name="isc-config"></a> file necessary for launching this experiment on the isc. 
   The key information included in the ISC Config file is as follows.
    - `isc_project_id`: The ID of the ISC Project created at [**Step 7**](#isc-project-id) above.
    - `experiment_name`: This name will appear in the **Experiments Table**. Use this name to uniquely identify this 
        experiment from your other experiments at a glance, for example by encoding hyper-parameters that you are testing.
    - `gpu_type`: The type of GPU that you are requesting for your experiment. At this time, the only supported 
        `gpu_type` is **`24GB VRAM GPU`**, so leave this unchanged.
    - `nnodes`: This is the number of nodes that you are requesting. Each node will have a number of GPUs. The **Rapid 
        Cycling** cluster nodes each have **6 GPUs**, and there are currently a **maximum of 12 nodes available**, so be 
        sure to request between 1 and 12 nodes.
    - `output_path`: This is the directory where the outputs of your experiment will be saved, including reports from 
        each node utilised by your experiment. It can be helpful to consider how you would like to structure your 
        output(s) directory to keep your experiment results organised (i.e. subdirectories for experiment groups).
    - `command`: This is the **CMD** supplied when running your docker container. In this `fashion_mnist` example we
      demonstrate chaining the suitable instructions to run a **torchrun** command. We `source ~/.bashrc` to configure
      the nodes to communicate over infiniband (note this will soon not be required), activate the virtual environment
      with `source ~/.fashion/bin/activate` and then launch torchrun with `torchrun --nnodes=10 --nproc-per-node=6 ...`.
      When launching torchrun it is important to specify the torchrun `--nnodes` argument equal to the `nnodes` argument
      above.

After you have run `prep_data.py` (above) to pre-download the Fashion MNIST dataset, you can launch this experiment to 
train on the ISC using the following command.

```bash
cd ~/isc-demos/fashion_mnist
isc train fashion_mnist.isc
isc experiments # view a list of your experiments
```

When you launch an experiment, the ISC will **commit and push** your container to our private docker registry, and then
run your container once on each node with the command provided in the `.isc` file.

You should recieve the response `Success: Experiment created`. Running `isc experiments` you should be able to see your 
experiment in the experiments table with the status `enqueued`. Other details about your experiment displayed include 
the following.
 - `ID`: The UUID assigned to uniquely idenfity your experiment in our database.
 - `Name`: The experiment name you specified in the `.isc` file, `fashion_mnist` in this case.
 - `NNodes`: The number of nodes requested for your experiment.
 - `Output Path`: The path to your experiment output directory. It may be necessary to zoom out to see the full path.
 - `Status`: The current status of your experiment.

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

You can launch a tensorboard instance to track the training and performance metrics of the experiment with the following 
command (**Note:** this assumes you are accessing the ISC from an IDE such as VSCode which does automatic port-forwarding. 
See below for instructions on how to view tensorboard when accessing the ISC from terminal without IDE).

```bash
tensorboard --logdir <Output Path from ISC Experiments table> --port <port>
```

Tensorboard will attempt to launch on a default port (typically 6006) if `--port` is not specified. You can then view the 
tensorboard at `http://localhost:<port>/`. Tensorboard recursively searches for tensorboard event logs in the directory 
passed after the `--logdir` flag, so it will discover the event logs in the `/tb` subdirectory.

![fashion_mnist_tensorboard](https://github.com/StrongResearch/isc-demos/blob/main/fashion_mnist/fashionmnist_tensorboard.png?raw=true)

If you are accessing the ISC from a terminal (not an IDE such as VSCode) then you will need to manually forward the port 
that tensorboard is served on by extending the tensorboard launch command as follows.
```bash
tensorboard --logdir <Output Path from ISC Experiments table> --port <port>  --host 0.0.0.0
```
Tensorboard can then be viewed by entering `192.168.127.70:<port>` in the address bar of your browser.

Continue to track the progress of your experiment while it cycles by checking in on the `rank_0.txt` file and the 
tensorboard. You can also view the contents of the `rank_0.txt` file by visiting the **Experiments** tab on 
**Control Plane** (https://cp.strongcompute.ai/) and clicking on the **Logs** button, or cancel the experiment
by clicking on the **Cancel** button.

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

## 3. Data Parallel Scaling <a name="data-parallel-scaling"></a>
When scaling to more GPUs, it is important to consider the impact this will have on your model training. 

One important thing to consider is the potential change in effective batch size. 

`effective_batch_size = n_gpus * batch_size_per_gpu`

Two common approaches to this are as follows:
1. **Maintain the original learning rate as well as the original effective batch size**

To achieve this, you would need to lower the batch size per GPU. For example, if you are scaling from 32 GPUs to 64, halve the batch size per GPU.

2. **Scale the original learning rate to the new increased effective batch size**

With increased effective batch size there is an opportunity to increase the learning rate to take advantage of the more stable gradient. In general, experimentation is required to determine the optimal increased learning rate. In our experience, a good starting heuristic is to increase the learning rate by the square root of the ratio of the new effective batch size to the original effective batch size.

For example, when scaling from an effective batch size of 32 to 128, the suggested new learning rate can be calculated as follows.

`new_learning_rate = sqrt(128/32) * original_learning_rate` 

## 4. Uploading your dataset <a name="data-transfer"></a>
Strong Compute currently supports uploading private datasets 100GB or less in size from AWS S3 buckets.
1. Visit the **Datasets** page on Control Plane (ensure you have selected the intended Organisation from the menu top-right) and click on **New Dataset**
2. Complete the New Dataset form, including AWS S3 bucket name and credentials information, leave the format set to "small_unstructured", and click on **Add Dataset**.
3. Returning to the **Datasets** page you will see your dataset registered on Control Plane. The status of the new dataset will report its progress through the preprocessing pipeline including **Created** indicating the dataset has been added to the web application, but has not yet been cached onto the ISC system, **Caching** indicating the dataset is currently being cached from S3 (this may take several minutes for larger S3 buckets), and **Available** indicating the dataset has been cached onto the ISC system, and can be accessed in training. Note the `dataset_id` for the new dataset for use in the next step.
4. Edit your `.isc` file to include the additional field `dataset_id="<dataset_id>"`. The ISC will mount your dataset **read-only** at `/data/<contents-of-your-bucket>` which your cont.

**Note:** All users within the Organisation that the Dataset was created under will be able to see and access this dataset. Currently datasets created in the above fashion are only accessible in training, not from the ISC Portal.
