from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision.datasets import MNIST, FashionMNIST, SVHN
from torchvision import transforms
import torchvision
import numpy as np
from scipy.stats import ortho_group


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # self.train=train

    def train_dataloader(self):
        transform = transforms.Compose([
            torchvision.transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ])
        dataset = MNIST('./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, shuffle=True)
        return loader
    
    def test_dataloader(self):
        transform = transforms.Compose([
            torchvision.transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ])
        dataset = MNIST('./data', train=False, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)
        return loader

class SVHNDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ])
        dset = SVHN(root='./data', download=True, transform=transform)
        torch.manual_seed(0)
        train_len = int(0.7*len(dset))
        self.train_dset, self.test_dset = torch.utils.data.random_split(dset, [train_len, len(dset)-train_len])

    def train_dataloader(self):
        loader = DataLoader(self.train_dset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, shuffle=True)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)
        return loader
    
class FMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        transform = transforms.Compose([
            torchvision.transforms.Resize(32),
            transforms.ToTensor(),
        ])
        dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, shuffle=True)
        return loader
    
class GaussianDataset(Dataset):
    
    def __init__(self, n_samples, m=1024, r=0.025, transform=None, conv=False):
        # m = 20
        # r = 0.25
        # sigmas = 2*np.exp(-r*np.arange(m))
        # m = 1024
        # r = 0.025
        sigmas = 2*np.exp(-r*np.arange(m))
        self.transform = transform
        # self.X = sigma*torch.randn((n_samples, dimension[0], dimension[1])) + mu
        np.random.seed(seed=233423)
        U = ortho_group.rvs(m)
        # U = np.dot(u, u.T)
        # cov_mat = np.diag(sigmas**2)
        cov_mat = U @ np.diag(sigmas**2) @ U.T
        self.X = np.random.multivariate_normal(np.zeros(sigmas.shape[0]), cov_mat, n_samples)
        self.X = torch.tensor(self.X).float()
        # if conv is False:
        #     self.X = torch.flatten(self.X, start_dim=2, end_dim=3).squeeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(0)

class GaussianDataModule(LightningDataModule):
    def __init__(self, batch_size, m, r):
        super().__init__()
        self.batch_size = batch_size
        self.m = m
        self.r = r
        
    def train_dataloader(self):
        trainset = GaussianDataset(n_samples=50000, m=self.m, r=self.r)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)
        return train_loader

class Sawbridge(LightningDataModule):
    def __init__(self, batch_size, n=10000, n_sample=1024):
        super().__init__()
        self.n_sample = n_sample
        self.batch_size = batch_size
        t = torch.linspace(0, 1, n_sample)
        torch.manual_seed(123)
        U = torch.rand((n,1))
        X = t - (t >= U).float()
        X = X.unsqueeze(2).unsqueeze(2)
        y = torch.zeros(X.shape[0])
        dataset = torch.utils.data.TensorDataset(X, y)
        len_keep = int(0.8*len(dataset))
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [len_keep, len(dataset) - len_keep])

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, shuffle=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=10000, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

