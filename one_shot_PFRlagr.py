import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST,MNIST, CIFAR10
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import dataloaders
import tqdm
import huffman
import models

from NERDlagr import GenRD as GenRDlagr
# device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# print(device)

def est_dist(X, Y):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    return torch.mean(torch.linalg.norm(X-Y, dim=1)**2)

def zipf_codebook(N, R):
    lam = 1 + 1/(R + np.log2(np.e)/np.e +1)
    huff_weights = dict()
    for k in range(1, N+1):
        huff_weights[k] = k**(-lam)
    huffman_codebook = huffman.codebook(huff_weights.items())
    return huffman_codebook
    
def est_rate_zipf(Ks, codebook):
    Ks = Ks + 1
    l = 0
    for K in Ks:
        l += len(codebook[K])
    return l / len(Ks)

def est_rate_ent(Ks):
    ua,uind=np.unique(Ks,return_inverse=True)
    hist = np.bincount(uind)
    hist = hist/sum(hist)
    # print(hist)
    return -np.inner(hist, np.log2(hist))

def squared_distances(x, y):
    x = x.reshape(x.shape[0], -1).unsqueeze(0)
    y = y.reshape(y.shape[0], -1).unsqueeze(0)
    C = torch.cdist(x, y).squeeze()
    return C**2
    
def invDR_batch(x, y, lam_opt):
    denom = torch.exp(lam_opt*squared_distances(x, y))+1e-9
    return 1 / denom

# def compress_PFR_batch(x_batch, model, beta, N):
#     #generate random codebook
#     with torch.no_grad():
#         z = torch.randn(N, model.latent_dim).to(device)
#         Ys = model.generator(z)
#         Ts = torch.tensor(np.cumsum(np.random.exponential(1, N))).to(device)
#         DRs = invDR_batch(x_batch, Ys, beta)
#         Ks = DRs*Ts[None,:]
#         K_batch = torch.argmin(Ks, dim=1)
#         return K_batch, Ys[K_batch], x_batch

def compress_PFR_batch(x_batch, model, beta, N):
    #generate random codebook
    with torch.no_grad():
        z = torch.randn(N, model.latent_dim).to(device)
        Ys = model.generator(z)
        weighted_exp = np.random.exponential(1, N)
        Ts = torch.tensor(np.cumsum(weighted_exp)).to(device)
        Ks = get_Ks(x_batch, Ys, beta, Ts[None,:])
        K_batch = torch.argmin(Ks, dim=1)
        return K_batch, Ys[K_batch], x_batch

def get_Ks(x, y, lam_opt, Ts):
    return torch.log(Ts) - lam_opt*squared_distances(x,y)

def compress_ORC_batch(x_batch, model, beta, N):
    with torch.no_grad():
        z = torch.randn(N, model.latent_dim).to(device)
        Ys = model.generator(z)
        weights = np.array([N/(N-j) for j in range(N)])
        weighted_exp = weights * np.random.exponential(1, N)
        Ts = torch.tensor(np.cumsum(weighted_exp)).to(device)
        Ks = get_Ks(x_batch, Ys, beta, Ts[None,:])
        K_batch = torch.argmin(Ks, dim=1)
        return K_batch, Ys[K_batch], x_batch
        
    
def calc_RD_PFR(loader, model,R, beta, N):
    Rate = 0
    Rate_ent = 0
    Dist = 0
    cbook = zipf_codebook(N, R)
    with torch.no_grad():
        for x,_ in loader:
            K, Y, _ = compress_PFR_batch(x.to(device), model, beta, N)
            Dist += est_dist(x.to(device), Y).item()
            Rate += est_rate_zipf(K.cpu().numpy(), cbook)
            Rate_ent += est_rate_ent(K.cpu().numpy())
    return Rate/len(loader), Rate_ent/len(loader), Dist/len(loader)

def calc_RD_ORD(loader, model,R, beta, N):
    Rate = 0
    Rate_ent = 0
    Dist = 0
    cbook = zipf_codebook(N, R)
    with torch.no_grad():
        for x,_ in loader:
            K, Y, _ = compress_ORC_batch(x.to(device), model, beta, N)
            Dist += est_dist(x.to(device), Y).item()
            Rate += est_rate_zipf(K.cpu().numpy(), cbook)
            Rate_ent += est_rate_ent(K.cpu().numpy())
    return Rate/len(loader), Rate_ent/len(loader), Dist/len(loader)


def calc_RDlagr(loader, model, lmbda_dual):
    """Evaluates R(D) for trained Q_Y at distortion=D."""
    Rate = 0
    Dist = 0
    model.to(device)
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            z = torch.randn(30000, model.latent_dim).to(device)
            y = model.generator(z).to(device)
#             print(x.device, y.device)
            dist_mat = model._squared_distances(x, y)
            log_mu_x = torch.logsumexp(lmbda_dual*dist_mat, dim=1) - np.log(dist_mat.shape[1])
            g_loss = -torch.mean(log_mu_x) / np.log(2)

            log_f_xy = torch.log(dist_mat)+lmbda_dual*dist_mat - log_mu_x[:,None]
            D = torch.mean(torch.exp(log_f_xy))
            R = (lmbda_dual*D / np.log(2)) + g_loss
            Rate += R.item()
            Dist += D.item()
    return (Rate)/len(loader), Dist/len(loader), lmbda_dual

def calc_RD_curve(lmbdas, loader, N=1000, data_name="MNIST"):

    rates_true_rd = []
    dists_true_rd = []
    rates_PFR_zipf = []
    rates_PFR_ent = []
    rates_PFR_UB = []
    dists_PFR = []

    rates_ORD_zipf = []
    rates_ORD_ent = []
    rates_ORD_UB = []
    dists_ORD = []

    for lam in tqdm.tqdm(lmbdas):
        if data_name == "SVHN":
            generator = models.Generator(img_size=(32,32,3), latent_dim=128, dim=32)
            model = GenRDlagr(D=lam, data_name=data_name, generator=generator)
            model.latent_dim = args.latent_dim
        elif data_name == "Gaussian":
            m = 20
            generator = models.Decoder_FC(m, 100)
            model = GenRDlagr(D=lam, data_name=data_name, generator=generator)
            model.latent_dim = 100
        # model = GenRDlagr(lmbda=lam)
        checkpoint = torch.load(f'trained_lagr/trained_{data_name}_L2/NERD_{data_name}_lmbda{lam:.3f}.pt')
        model.load_state_dict(checkpoint)
        model.to(device)
        r, d, _ = calc_RDlagr(loader, model, lam)

        # r,d,_ = calc_RD(loader, model, D)

        rates_true_rd.append(r)
        dists_true_rd.append(d)
        rate_ord_zipf, rate_ord_ent, dist_ord = calc_RD_ORD(loader, model, r, lam, N)
        rates_ORD_zipf.append(rate_ord_zipf)
        rates_ORD_ent.append(rate_ord_ent)
        dists_ORD.append(dist_ord)

        rate_pfr_zipf, rate_pfr_ent, dist_pfr = calc_RD_PFR(loader, model, r, lam, N)
        rate_pfr_UB = r + np.log2(r+1)+5
        rates_PFR_zipf.append(rate_pfr_zipf)
        rates_PFR_ent.append(rate_pfr_ent)
        rates_PFR_UB.append(rate_pfr_UB)
        dists_PFR.append(dist_pfr)


        print(f'r={r:.4f},r_pfr_zipf={rate_pfr_zipf:.4f},r_ord_zipf={rate_ord_zipf:.4f}, d={d:.4f},d_pfr={dist_pfr:.4f}, d_ord={dist_ord:.4f}, lam={lam:.4f}')

    print(f'rates={rates_true_rd}')
    print(f'rates_ORD_zipf={rates_ORD_zipf:}')
    # print(f'rates_ORD_ent={rates_ORD_ent:}')    
    print(f'rates_PFR_zipf={rates_PFR_zipf}')
    # print(f'rates_PFR_ent={rates_PFR_ent}')
    print(f'rates_PFR_UB={rates_PFR_UB}')
    # print(f'DD={DD}')
    print(f'dists_alt={dists_true_rd}')
    print(f'dists_PFR={dists_PFR}')
    print(f'dists_ORD={dists_ORD}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--lmbdas', type=float, nargs='+', action='append', help='Distortion points')
    parser.add_argument('--N', type=int, default=100000, help='num samples for random codebook')
    # parser.add_argument('--M', type=int, default=20000, help='num samples for estimating MI')
    parser.add_argument("--data_name", type=str, default="MNIST", help="dataset name")
    args = parser.parse_args()
    print('hello')
    
    if args.data_name == "MNIST":
        dm = dataloaders.MNISTDataModule(100)
        loader = dm.test_dataloader()
    elif args.data_name == 'SVHN':
        dm = dataloaders.SVHNDataModule(100)
        loader = dm.test_dataloader()
    elif args.data_name == 'Gaussian':
        m = 20
        r = 0.25
        dm = dataloaders.GaussianDataModule(100, m, r)
        loader = dm.train_dataloader()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(args.lmbdas)
    calc_RD_curve(args.lmbdas[0], loader, args.N, args.data_name)
