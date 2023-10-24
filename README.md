# ISC Demos

Welcome to the Strong Compute ISC Demos repo, here you will find all the instructions you need to get set up to train 
Pytorch models on the Strong Compute ISC.

## Getting Started

### 1. Setting up the VPN
Before connecting to the Strong Compute ISC, you must have recieved login credentials from Strong Compute by email. 
Please reach out to us if you have not received this email.

#### For MacOS and Windows

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

#### For Linux

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

### 2. Creating your ISC User and Organisation credentials
Now that you have your VPN correctly installed, configured, and enabled, we'll get you set up with a login to the ISC 
associated with your User and Organisation.

Your User login to the ISC will be associated with an Organisation. If you are the first person in your organisation to 
register, then you will be assigned to a brand new Organisation, and you can then invite others to this organisation. If 
someone has already created an Organisation on Control Plane that you should be associated with, they can send you an 
invitation to join (step 2 below) and you can skip to step 3 below. Check with your colleagues before proceeding to 
avoid doubling-up Organisations.

1. Visit the Strong Compute Control Plane at https://cp.strongcompute.ai/, click on "Register", register with your email 
    and choose a suitable password.
2. If you are the first person in your Organisation to register, click on the menu at the top right of the page 
    (labelled with your email and Organisation name) and click on "Organisations". On the "Your Organisations" page, 
    click on the tile for your organisation. When you first register, your Organisation will be named 
    `<your email>'s Personal Organisation`. From the "Organisation Settings" page you can update the name of your 
    Organisation, view current members, and send invitations to new members. If you would like to use Control Plane to 
    track your usage on Azure, AWS, GCP, or Lambda you can also manage your API keys for these platforms, though this is 
    not necessary for using the ISC.
2. Click on the menu at the top right of the page and click on "Settings".
3. At the bottom of the page, click on "NEW SSH KEY".
4. You will need to generate an RSA key pair by opening terminal and running `ssh-keygen`. When prompted, provide a 
    filename in which to save the key. You can also optionally enter a passphrase (or just press enter twice). This will 
    generate two separate files, containing your public and private keys. The public key is saved in the file ending in `.pub`.
5. Open the file containing the public key. The public key file contents should start with `ssh-rsa` and end with 
    `.local`. Copy the entire contents of this file and paste into the input field on Control Plane beneath "Public key 
    contents", then click "SUBMIT NEW SSH PUBLIC KEY". Note, only RSA public keys are currently supported, reach out to 
    us if this is unsuitable for you.
6. When you are returned to the Settings page on Control Plane, click on "NEW API KEY". Optionally name your API key or 
    leave this input field blank. If you already have multiple Organisations established, select the Organisation to 
    associate this API key with. Click on "GENERATE NEW API ACCESS TOKEN". You will be presented with the API Access 
    Token associated with your API key. Save a copy of this API Access Token. Be careful to save the entire API Access 
    Token, noting that some characters will be obscured due to the fixed width of the display window. For security 
    purposes you will only be shown this API Access Token once and will not be able to access it again, though you can 
    always create a new API Key.
7. Click on "Back to Settings". You should see the new API Key that you just created, and an associated SSH Username. 
    You will be able to use this SSH Username to connect to the ISC via SSH.
8. Open terminal and enter the entire SSH Username into the command line. The command should start with `ssh` and end 
    with `@<ip-address>`. You should be greeted by the Strong Compute logo and ISC welcome message. This indicates that 
    you have successfully logged into your home directory on the ISC.
9. To create your required `credentials.isc` file, run `isc login` and enter the API Key you saved previously at step 6. 
    This credentials file is used to authenticate you when you launch experiments on the ISC, and should be saved in the 
    root of your home directory. Run `isc ping` and you should receieve `Success: {'data': 'pong'}` in response to 
    indicate that your credentials file has been created correctly.
10. Create a virtual environment by running `python3 -m virtualenv ~/.venv` and activate your virtual environment by 
    running `~/.venv/bin/activate`. You will need to ensure that you have activated your virtual environment whenever 
    you launch experiments on the ISC.

Congratulations, you are all set to start running experiments on the ISC. Follow the next steps in this guide to 
configure and launch your first "hello world" experiment, and learn about necessary steps to make sure your experiment 
is "interruptable" (including what this means).

## Interruptible Experiments

We will now inspect and run some code to prepare and launch an experiment on the ISC. This example will demonstrate the 
principle and application of interruptibility, which will be important to consider when developing your experiments to 
run successfully on the ISC.

### Rapid Cycling and Burst To Cloud

The ISC is comprised of a Rapid Cycling stage, and a Burst To Cloud ("Burst") stage. Experiments launched on the ISC are 
run first in Rapid Cycling as a validation step before being Burst to the cloud. This provides Users with near-immediate 
feedback on the viability of their code before their experiment is launched in a cloud environment and incurs costs.

**Action:** Run `isc experiments` to see a table report of all of your historic experiments. When you first register, 
this table will be empty. Each time you launch an experiment on the ISC, a record of that experiment will appear in this 
table. Only your experiments will be visible to you in this table. There may be other experiments scheduled to run on 
the ISC at the same time that will not be visible to you.

Experiments share time on the Rapid Cycling cluster by means of a queue which is cycled at fixed time intervals. The 
queue is comprised of two sub-queues, indicated by status `enqueued` and `paused` respectively. All `enqueued` 
experiments are cycled before any `paused` experiment is cycled.

When a new experiment is launched onto the Rapid Cycling cluster, it joins the `enqueued` queue. While the experiment is 
running, its status will be recorded as `running`, and it will be permitted to run for a fixed period of **90 seconds**. 
If the experiment fails during its running cycle (for example if there was an error in the experiment code) then the 
experiment status will be recorded as `failed` and the experiment will be removed from the queue. Otherwise, at the end 
of this period, the experiment will be paused to allow other experiments to cycle, and it will join the `paused` queue. 
Experiments from the `paused` queue are cycled when there are no experiments waiting in the `enqueued` queue. 

Once an experiment has **completed 5 cycles** it will be scheduled for Burst To Cloud. This means the experiment will be 
assigned a dedicated cluster in the cloud, and will be launched on this dedicated cluster to train. In case of cloud 
interruption (for example due to or hardware failure), another dedicated cluster will be assigned and the experiment 
resumed on the new dedicated cluster.

The ability to abruptly interrupt experiments or "interruptibility" is crucial for the purpose of Rapid Cycling, pausing 
and resuming. The main approach to achieve interruptibility is robust and regular checkpointing which we will 
demonstrate with the example project that follows.

### Hello World with Fashion MNIST

To follow this demonstration, first ensure you have activated your virtual environment, cloned this repo in your home 
directory on the ISC, and installed the necessary requirements with the following commands. Only Pytorch models are 
currently supported on the ISC and distributed training is coordinated using torchrun. Therefore, of the requirements 
included in the `requrements.txt` file, it is mandatory to have Pytorch (2.0.1 or later) installed at a minimum. Other 
requirements included in the `requrements.txt` file are necessary for one or more of the other example projects 
showcased in this repo.

```bash
cd ~
source ~/.venv/bin/activate
git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos
pip install -r requirements.txt
```

You will also need to clone the `cycling_utils` and install it as a package in editable mode with the following 
commands. The `cycling_utils` package contains helpful functions and classes for achieving interruptibility in 
distributed training. Installing `cycling_utils` in editable mode will allow you to extend this package at any time with 
your own modules as need be without needing to reinstall the package.

```bash
cd ~
git clone https://github.com/StrongResearch/cycling_utils.git
pip install -e cycling_utils
```

Next return to the `isc-demos` repo directory and inspect the contents of the `fashion_mnist` subdirectory.

```bash
cd ~/isc-demos/fashion_mnist
ls
```

The `fashion_mnist` subdirectory contains the following files of interest.
1. `prep_data.py` includes commands for downloading the required dataset (Fashion MNIST). By running 
    `python -m prep_data` we download the dataset to the `fashion_mnist` directory, from which it is available to the 
    Rapid Cycling cluster, which will mean that this data is ready to go when the experiment is launched and no cycling 
    time is wasted waiting for the data to download.
2. `model.py` includes a description of the neural network model that we will train. 
3. `train.py` describes configuration for distributed training, initialisation, and distributed training loops. Take a 
    few minutes to read and understand the contents of this file, there are lots of notes to explain what's happening 
    and feel free to reach out with any questions. Note that `train.py` provides for command line arguments to be 
    passed, we will see how when looking at the next file.
4. `fashion_mnist.isc` is the file necessary for launching the experiment on the isc. The necessary information to 
    include in the `.isc` file is as follows.
    - `experiment_name`: Use this name to uniquely identify this experiment from your other experiments.
    - `gpu_type`: The type of GPU that you are requesting for your experiment. At this time, the only supported 
        `gpu_type` is `24GB VRAM GPU`, so leave this unchanged.
    - `nnodes`: This is the number of nodes that you are requesting. Each node will have a number of GPUs. The Rapid 
        Cycling cluster nodes each have 6 GPUs, and there are currently 12 nodes available, so be sure to request 
        between 1 and 12 nodes.
    - `venv_path`: The path to your virtual environment which we set up above. If you followed the instructions above 
        for "Creating your ISC User and Organisation credentials" then you should be able to leave this unchanged as 
        well.
    - `output_path`: This is the directory where the outputs of your experiment will be saved, including reports from 
        each node being used by your experiment. It can be helpful to consider how you would like to structure your 
        output(s) directory to keep your experiment results organised.
    - `command`: This is the launch command passed to torchrun to commence your experiment. Note the call to `train.py` 
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
 - `ID`: UUID assigned to uniquely idenfity your experiment in our database.
 - `Name`: The experiment name you specified in the `.isc` file, `fashion_mnist` in this case.
 - `NNodes`: The number of nodes requested for your experiment.
 - `Output Path`: The path to your experiment output directory. It may be necessary to zoom out to see the full path.

 When you see your experiment transition from `enqueued` to `running`, navigate to the experiment output directory. You 
 will find a number of `rank_X.txt` files, corresponding to each of the `NNodes` nodes requested for your experiment. 
 You can thus verify that all requested nodes were initialised. Each `rank_X.txt` file should contain at least the 
 following. By default, only the `rank_0.txt` file should contain anything further.

 ```bash
 WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being \
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
 ```

 **Action:** Open the `rank_0.txt` file and inspect the contents. Compare the outputs in this file with the reporting 
 points within `train.py`. If there are errors in your code, the `rank_X.txt` files will be where to look to understand 
 what has gone wrong and how to fix it.

 Returning to the command line, you can launch a tensorboard instance to track the training and performance metrics of 
 the experiment with the following command.

 ```bash
 tensorboard --logdir <Output Path from ISC Experiments table>
 ```

Congratulations, you have successfully launched your first experiment on the ISC!

## More examples

The following examples further demonstrate how to implement interruptibility in
distributed training scripts using checkpointing, atomic saving, and
stateful samplers.

These examples are being actively developed to achieve (1) interruptibility
in distributed training, (2) verified completion of a full training run, and
(3) achievement of benchmark performance published by others (where applicable). 
Each example published below is annotated with its degree of completion. Examples
annotated with [0] are "coming soon".

### Hello World

| Title          | Description | Model   | Status        |
| :---           |    :----:   |:----:|          ---: |
| Fashion MNIST  | Title       || Here's this   |
| CIFAR100       | Text        || And more      |

- [cifar100-resnet50.isc](./cifar100-resnet50/cifar100-resnet50.isc) [3]
- [fashion_mnist.isc](./fashion_mnist/fashion_mnist.isc) [3]
- WIP [dist_model_parallel.isc](./dist_model_parallel.isc) [0]

### pytorch-image-models (timm)

(from https://github.com/huggingface/pytorch-image-models)

- [resnet50.isc](./pytorch-image-models/resnet50.isc) [2]
- [resnet152.isc](./pytorch-image-models/resnet152.isc) [2]
- [efficientnet_b0.isc](./pytorch-image-models/efficientnet_b0.isc) [2]
- [efficientnet_b7.isc](./pytorch-image-models/efficientnet_b7.isc) [2]
- [efficientnetv2_s.isc](./pytorch-image-models/efficientnetv2_s.isc) [2]
- WIP [efficientnetv2_xl.isc](./pytorch-image-models/efficientnetv2_xl.isc) [2]
- [vit_base_patch16_224.isc](./pytorch-image-models/vit_base_patch16_224.isc) [2]
- WIP [vit_large_patch16_224.isc](./pytorch-image-models/vit_large_patch16_224.isc) [2]

### tv-segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

- WIP [fcn_resnet101.isc](./tv-segmentation/fcn_resnet101.isc) [1]
- WIP [deeplabv3_mobilenet_v3_large.isc](./tv-segmentation/deeplabv3_mobilenet_v3_large.isc) [1]

### tv-detection

(from https://github.com/pytorch/vision/tree/main/references/detection)

- WIP [maskrcnn_resnet50_fpn.isc](./tv-detection/fasterrcnn_resnet50_fpn.isc) [0]
- WIP [retinanet_resnet50_fpn.isc](./tv-detection/retinanet_resnet50_fpn.isc) [0]

## Detectron2

(from https://github.com/facebookresearch/detectron2)

- WIP [detectron2.isc](./detectron2.isc) [0]
- WIP [detectron2_densepose.isc](./detectron2_densepose.isc) [0]

## Large Language Models

- WIP [llama2.isc](./llama2.isc) [0]
- WIP [mistral.isc](./mistral.isc) [0]
