# MONAI Generative Models Installation
For this demonstration, you will need to clone the MONAI GenerativeModels GitHub repository and follow the instructions for installation. This will install the `generative` package from MONAI.
You will then need to run `pip install -r requirements-dev.txt` in this directory to install other necessary dependencies. You may then also need to ensure that monai version 1.2.0 is installed using the command `pip install monai==1.2.0` as later versions of monai do not support all of the transforms used in this example.

In [../site-packages/generative-0.2.2-py3.10.egg/generative/losses/perceptual.py] on line 206 change from:
`norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))`
to
`norm_factor = torch.sqrt(torch.sum(x**2 + eps, dim=1, keepdim=True))`