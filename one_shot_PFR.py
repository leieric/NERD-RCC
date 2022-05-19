import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST,MNIST, CIFAR10
from torch.utils.data import DataLoader
import tqdm
import huffman

from rd_primal_minmax_bisection import GenRD
from rd_primal_minmax_FMNIST_bisection import GenRD as GenRD_FMNIST
from rd_primal_minmax_CIFAR_bisection import GenRD as GenRD_CIFAR
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
        ])
dataset = FashionMNIST('.', train=True, download=True, transform=transform)
n = 5000
subset = torch.utils.data.Subset(dataset, list(range(n)))
loader = DataLoader(subset, batch_size=n, shuffle=True)

dataset = MNIST('.', train=False, download=True, transform=transform)
test_loader = DataLoader(subset, batch_size=100, shuffle=True)

def est_dist(X, Y):
    return torch.mean(torch.linalg.norm(X-Y, dim=(1,2,3))**2)

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

def calc_RD(loader, model, D):
    Rate = 0
    Dist = 0
    model.to(device)
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            z = torch.randn(x.shape[0], model.latent_dim).to(device)
            y = model.generator(z).to(device)
#             print(x.device, y.device)
            C = model._squared_distances(x,y)
            beta = model.inner_max(C)
            # print(beta)
            R = beta*D - torch.mean(torch.log(torch.mean(torch.exp(beta*C), dim=1)+1e-14))
            # D = torch.mean(C*torch.exp(beta*C) / (torch.mean(torch.exp(beta*C), dim=1)[:,None])+1e-14)
            D = torch.mean(C*torch.exp(beta*C) / ((torch.mean(torch.exp(beta*C), dim=1)+1e-14)[:,None]))
            Rate += R.item()
            Dist += D.item()
    return (Rate/np.log(2))/len(loader), Dist/len(loader), beta

DD = [70, 60, 50, 40, 30, 20, 17.5, 15, 12.5, 10, 5]
# DD = [55, 50, 45, 40, 35, 30, 25]
DD.reverse()
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

for D in DD:
    model = GenRD(D=D)
    checkpoint = torch.load(f'trained/trained_genRD_minmax_FMNIST/GenRD_trained_genRD_minmax_FMNIST_D{D:.3f}.pt')
    model.load_state_dict(checkpoint)
    model.to(device)
    r, d, beta = calc_RD(loader, model, D)
    
    # r,d,_ = calc_RD(loader, model, D)
    
    rates_true_rd.append(r)
    dists_true_rd.append(d)
    
    N = 100000
    
    rate_ord_zipf, rate_ord_ent, dist_ord = calc_RD_ORD(loader, model, r, beta, N)
    rates_ORD_zipf.append(rate_ord_zipf)
    rates_ORD_ent.append(rate_ord_ent)
    dists_ORD.append(dist_ord)
    
    rate_pfr_zipf, rate_pfr_ent, dist_pfr = calc_RD_PFR(loader, model, r, beta, N)
    rate_pfr_UB = r + np.log2(r+1)+5
    rates_PFR_zipf.append(rate_pfr_zipf)
    rates_PFR_ent.append(rate_pfr_ent)
    rates_PFR_UB.append(rate_pfr_UB)
    dists_PFR.append(dist_pfr)
    
    
    print(f'r={r:.4f},r_pfr_zipf={rate_pfr_zipf:.4f},r_ord_zipf={rate_ord_zipf:.4f}, d={D:.4f},d_pfr={dist_pfr:.4f}, d_ord={dist_ord:.4f}, beta={beta:.4f}')

print(f'rates={rates_true_rd}')
print(f'rates_ORD_zipf={rates_ORD_zipf:}')
print(f'rates_ORD_ent={rates_ORD_ent:}')
print(f'rates_PFR_zipf={rates_PFR_zipf}')
print(f'rates_PFR_ent={rates_PFR_ent}')
print(f'rates_PFR_UB={rates_PFR_UB}')
print(f'DD={DD}')
print(f'dists_alt={dists_true_rd}')
print(f'dists_PFR={dists_PFR}')
print(f'dists_ORD={dists_ORD}')