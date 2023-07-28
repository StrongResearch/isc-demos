# nerf_ddp


This repo tries to reduce NeRF training time by applying Pytorch Distributed Data Parallel training (DDP). Much of the code is inspired by the original implementation by GitHub user [bmild](https://github.com/bmild/nerf) as well as PyTorch implementations from GitHub users [yenchenlin](https://github.com/bmild/nerf), [krrish94](https://github.com/krrish94/nerf-pytorch/) and the Medium Article (https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666). The code has been modified for correctness, clarity, and consistency.


There are two implementations (e.g., tiny-nerf and full nerf). It is good to start with tiny_nerf to understand the basic implementation of both NeRF and Pytorch DDP.


(1) tiny_nerf_ddp.py is modified from https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX and add Pytorch Data Parallel.

The author's repo is https://github.com/krrish94/nerf-pytorch


launch command if you have one machine with 8 GPUs
```
python tiny_nerf_pytorch_ddp.py -n 1 -g 8 -i 0
```



(1) nerf_ddp.py is modified from (https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666)ch


launch command if you have one machine with 8 GPUs
```
python nerf_pytorch_ddp.py -n 1 -g 8 -i 0
```


launch command if you have two machines with 8 GPUs each
```
python nerf_ddp.py -n 2 -g 8,8 -i 0
python nerf_ddp.py -n 2 -g 8,8 -i 1
```

The results show the with DDP, the NeRF training time can be reduced by 5 times.

