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

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import dataloaders

import matplotlib.pyplot as plt

# import model_resnet
import models


class WGANGP(LightningModule):

    def __init__(self,
                 latent_dim: int = 256,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, 
                 data_name = "MNIST",
                 img_size=(32,32,1),
                 dnn_size=64,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_name = data_name
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # networks
        self.discriminator = models.Discriminator(img_size, dnn_size)
        self.generator = models.Generator(img_size, self.latent_dim, dnn_size)

        self.validation_z = torch.randn((8, self.latent_dim))

        self.example_input_array = torch.zeros((2, self.latent_dim))

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0],1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn((imgs.shape[0], self.latent_dim))
        z = z.type_as(imgs)

        lambda_gp = 10

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
#             sample_imgs = self.generated_imgs[:6]
#             grid = torchvision.utils.make_grid(sample_imgs)
#             self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = -torch.mean(self.discriminator(self(z))) 
            with torch.no_grad():
                wass_obj = g_loss + torch.mean(self.discriminator(imgs))
            tqdm_dict = {'wass_obj': wass_obj.item()}
            self.log_dict(tqdm_dict, prog_bar=True)
            return g_loss

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake_imgs = self(z)

            # Real images
            real_validity = self.discriminator(imgs)
            # Fake images
            fake_validity = self.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) 
            tqdm_dict = {'wass_obj': -d_loss.item(), 'gp':gradient_penalty.item()}
            self.log_dict(tqdm_dict, prog_bar=True)
            return d_loss + lambda_gp * gradient_penalty

    def configure_optimizers(self):
        n_critic = 5

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=lr, betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )


    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        
        grid = torchvision.utils.make_grid(sample_imgs)
        grid = 0.5*(grid + 1)
#         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        if self.current_epoch % 10 == 0:
            plt.figure()
            plt.imshow(grid.detach().cpu().permute(1, 2, 0))
            plt.savefig(f'trained/figures_{self.data_name}/epoch{self.current_epoch}')


def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if args.data_name == "CIFAR":
        dm = dataloaders.CIFARDataModule(args.batch_size)
        args.img_size=(32,32,3)
        args.dnn_size=64
    elif args.data_name == "MNIST":
        dm = dataloaders.MNISTDataModule(args.batch_size)
        args.img_size=(32,32,1)
        args.dnn_size=32
    model = WGANGP(**vars(args))
    
    # print(args.init)
    if args.init==1:
        ckpt = torch.load(f'trained/trained_{args.data_name}/wgan_gp_{args.data_name}.ckpt')
        model.load_state_dict(ckpt)
    
#     checkpoint_callback = ModelCheckpoint(every_n_epochs=1, monitor='d_loss')
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=args.gpus[0], 
                      max_epochs=args.epochs,
#                      callbacks=[checkpoint_callback]
                     )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    
    trainer.fit(model, dm)
    torch.save(model.state_dict(), f"trained/trained_{args.data_name}/wgan_gp_{args.data_name}.ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
#     parser.add_argument("--gpus", type=int, default=[0], help="gpu list")
    parser.add_argument('-g','--gpus', type=int, nargs='+', action='append', help='gpu_list')
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="dimensionality of the latent space")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--init", type=int, default=0, help="init saved model")
    parser.add_argument("--data_name", type=str, default="MNIST", help="dataset")

    args = parser.parse_args()
    
    if not os.path.exists(f'trained/figures_{args.data_name}'):
        os.mkdir(f'trained/figures_{args.data_name}')
    if not os.path.exists(f'trained/trained_{args.data_name}'):
        os.mkdir(f'trained/trained_{args.data_name}')

    main(args)

