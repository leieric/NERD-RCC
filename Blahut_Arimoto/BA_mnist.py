import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from BlahutArimoto import run_BlahutArimoto
import ot

transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
        ])

# Choose Dataset
dataset = FashionMNIST('data/', train=True, download=True, transform=transform)
# Set sample size
n = 10000
subset = torch.utils.data.Subset(dataset, list(range(n)))
loader = DataLoader(subset, batch_size=n, shuffle=True)
# loader = DataLoader(dataset, batch_size=10000)

def run_BA(beta, loader):
    rate = 0
    distortion = 0
    n_nan = 0
    for x_pos,_ in loader:
        n,_,_,_ = x_pos.shape
        y_pos = x_pos + 0*torch.randn(x_pos.shape)
        M = ot.dist(x_pos.reshape(n, -1), y_pos.reshape(n, -1))
        p_x = (1/n)*torch.ones(n)
        R,D = run_BlahutArimoto(M, p_x.numpy(), beta, eps=1e-10)
    #     print(R, D)
#         if np.isnan(R) or np.isnan(D):
#             n_nan += 1
#         else:
        rate += R 
        distortion += D
    denom = len(loader) - n_nan
#     if len(loader) == 1:
#         denom = 1
    rate, distortion = rate/denom, distortion/denom
    return rate, distortion

betas = [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.13, 0.14, 0.2]
# betas = [0.25, 0.3]
# betas = [0.1]
rates = []
dists = []
for beta in betas:
    r,d = run_BA(beta, loader)
    print(f'beta={beta}, r={r}, d={d}')
#     r,d = run_BA_full(beta, dataset)
    rates.append(r)
    dists.append(d)
    
print(f'rates={rates}')
print(f'dists={dists}')
