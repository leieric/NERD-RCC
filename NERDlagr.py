"""
To run this template just do:
python wgan_gp.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# torch.autograd.set_detect_anomaly(True)

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pykeops.torch import LazyTensor

import matplotlib.pyplot as plt

import models
import scipy.optimize
import dataloaders

class GenRD(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0001,
                 batch_size: int = 64, 
                 lmbda: float=-1,
                 data_name="MNIST",
                 generator=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.eps = 1e-20

        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        # self.D = D
        self.lmbda_dual = lmbda
#         self.lmbda_dual = nn.Parameter(torch.tensor(init_lmbda)) #torch.ones(1, requires_grad=True)

        # networks
        self.generator = generator

        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)
        
        self.fname = data_name
        # print(self.D)
        self.figdir=f"trained_lagr/figures_{self.fname}/lmbda{self.lmbda_dual:2f}"
        if not os.path.exists(self.figdir):
            os.mkdir(self.figdir)
            # os.mkdir('trained/trained_{model.fname}')
        
        # self.len_dataloader = 0
        # self.epoch_loss = 0
        self.train_losses = []

    def forward(self, z):
        return self.generator(z)

    
    def _squared_distances(self, x, y, p=2):
        x = x.reshape(x.shape[0], -1).unsqueeze(0)
        y = y.reshape(y.shape[0], -1).unsqueeze(0)
        C = torch.cdist(x, y, p=p).squeeze()
        return C**2

    # def l1_distances(self, x, y):
    #     x = x.reshape(x.shape[0], -1).unsqueeze(0) #[1, n, nx]
    #     y = y.reshape(y.shape[0], -1).unsqueeze(0)
    
    def training_step2(self, batch, batch_idx):
        self.eps = (0.99**self.current_epoch)*self.eps
        x, _ = batch

        # sample noise
        z = torch.randn(70000, self.latent_dim)
        z = z.type_as(x)

        y = self(z)
        x = LazyTensor(x.view(x.shape[0], 1, -1))
        y = LazyTensor(y.view(1, y.shape[0], -1))
        # dist_mat = self._squared_distances(x, y)
        dist_mat = ((x-y)**2).sum(dim=2)
        # print(dist_mat.shape)
        c = 0
        # mean_z = torch.mean(torch.exp(self.lmbda_dual*dist_mat + c), dim=1)
        # log_mu_x = torch.logsumexp(self.lmbda_dual*dist_mat, dim=1) - np.log(dist_mat.shape[1])
        # log_mu_x = (self.lmbda_dual*dist_mat).exp().mean(dim=1).log()
        log_mu_x = (self.lmbda_dual*dist_mat).logsumexp(dim=1) - float(np.log(dist_mat.shape[1]))
        g_loss = -log_mu_x.mean(dim=0) / float(np.log(2))
        
        log_f_xy = dist_mat.log()+self.lmbda_dual*dist_mat + c - log_mu_x.view(x.shape[0], 1, -1)
        distortion = log_f_xy.exp().sum() / x.shape[0]#torch.mean(torch.exp(log_f_xy))
        rate = (self.lmbda_dual*distortion / float(np.log(2))) + g_loss
        
        # d_max = -torch.mean(torch.logsumexp(self.lmbda_dual*dist_mat, dim=1)) / np.log(2)

        
        
        # deriv = torch.mean(dist_mat*torch.exp(self.lmbda_dual*dist_mat) / ((torch.mean(torch.exp(self.lmbda_dual*dist_mat), dim=1)+1e-14)[:,None]))
        # denom_x = torch.mean(torch.exp(self.lmbda_dual*dist_mat), dim=1) + 1e-8
        # deriv = torch.mean(dist_mat*torch.exp(self.lmbda_dual*dist_mat) / denom_x[:,None])

        # tqdm_dict = {'rate': rate.item(), 'dual_var': self.lmbda_dual, 'distortion':distortion.item()}
        # self.log_dict(tqdm_dict, prog_bar=True)
        return g_loss

    def training_step(self, batch, batch_idx):
        self.eps = (0.99**self.current_epoch)*self.eps
        x, _ = batch

        # sample noise
        z = torch.randn(40000, self.latent_dim)
        z = z.type_as(x)

        y = self(z)
        # x = LazyTensor(x.view(x.shape[0], 1, -1))
        # y = LazyTensor(y.view(1, y.shape[0], -1))
        dist_mat = self._squared_distances(x, y, p=2)
        # dist_mat = ((x-y)**2).sum(dim=2)
        # print(dist_mat.shape)
        c = 0
        # mean_z = torch.mean(torch.exp(self.lmbda_dual*dist_mat + c), dim=1)
        log_mu_x = torch.logsumexp(self.lmbda_dual*dist_mat, dim=1) - np.log(dist_mat.shape[1])
        g_loss = -torch.mean(log_mu_x) / np.log(2)
        
        log_f_xy = torch.log(dist_mat)+self.lmbda_dual*dist_mat + c - log_mu_x[:,None]
        distortion = torch.mean(torch.exp(log_f_xy))
        rate = (self.lmbda_dual*distortion / np.log(2)) + g_loss
        
        d_max = -torch.mean(torch.logsumexp(self.lmbda_dual*dist_mat, dim=1)) / np.log(2)

        
        
        # deriv = torch.mean(dist_mat*torch.exp(self.lmbda_dual*dist_mat) / ((torch.mean(torch.exp(self.lmbda_dual*dist_mat), dim=1)+1e-14)[:,None]))
        # denom_x = torch.mean(torch.exp(self.lmbda_dual*dist_mat), dim=1) + 1e-8
        # deriv = torch.mean(dist_mat*torch.exp(self.lmbda_dual*dist_mat) / denom_x[:,None])

        tqdm_dict = {'rate': rate.item(), 'dual_var': self.lmbda_dual, 'distortion':distortion.item(), 'd_max':d_max.item()}
        # self.epoch_loss += g_loss.item()
        self.log_dict(tqdm_dict, prog_bar=True)
        return g_loss

            
            

    def configure_optimizers(self):
        
        b1 = 0.5
        b2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(b1, b2))
        return opt_g


    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        # grid = 0.5*(grid + 1)
#         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        # self.train_losses.append(self.epoch_loss / self.len_dataloader)
        # self.epoch_loss = 0
        if self.current_epoch % 25 == 0:
            plt.figure(1)
            plt.imshow(torch.clip(grid.detach().cpu().permute(1, 2, 0), 0, 1), cmap='gray')
            plt.savefig(f'{self.figdir}/epoch{self.current_epoch}')
            
#             plt.figure(2)
#             plt.clf()
#             plt.plot([loss if loss > 0 else 0 for loss in self.train_losses ])
# #             plt.ylim([-1, 100])
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.savefig(f'trained/{self.figdir}/training_loss')


def train_save(args, lmbda, generator, datamodule) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # model = GenRD(**vars(args))
    # print(D)

        
    model = GenRD(latent_dim=args.latent_dim,
                 lr=args.lr,
                 batch_size=args.batch_size,
                 lmbda=lmbda,
                 data_name=args.data_name,
                 generator=generator)
    if args.init_gan==1:
        # initialize with trained gan
        ckpt = torch.load(f'trained_gan/wgan_gp_{args.data_name}.ckpt')
        from wgan_gp import WGANGP
        model_gan = WGANGP(latent_dim=128, dnn_size=32, img_size=generator.img_size)
        model_gan.load_state_dict(ckpt)
        model.generator=model_gan.generator
    
    trainer = Trainer(accelerator='gpu',
                      devices=args.gpus[0], 
                      strategy='ddp',
                      max_epochs=args.epochs
                     )
    
    trainer.fit(model, datamodule)
    
    torch.save(model.state_dict(), f"trained_lagr/trained_{model.fname}/NERD_{model.fname}_lmbda{lmbda:.3f}.pt")

    
    



