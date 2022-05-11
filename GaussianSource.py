import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scipy.spatial
import scipy.special
import scipy.optimize
from scipy.stats import norm
import huffman

# rate-distortion functions

def lam_obj(lam, sigmas, D):
    ind_under = sigmas**2 <= lam
    lhs = lam * np.sum(1 - ind_under) + np.sum(sigmas[ind_under]**2)
    return lhs - D

def rev_wf(sigmas, D):
    # reverse waterfilling operation, return lambda
    if D > sum(sigmas**2):
        return max(sigmas**2)
    f_obj = lambda lam : lam_obj(lam, sigmas, D)
    lam_opt = scipy.optimize.bisect(f_obj, 0, D)
    return lam_opt

def rd_gaussian(D, sigmas):
    # gaussian R(D) with covariance singular values in sigmas
    lam_opt = rev_wf(sigmas, D)
    ind_over = sigmas**2 > lam_opt
    return np.sum(0.5*np.log2(sigmas[ind_over]**2 / lam_opt))

# Quantization functions
def dist_interval(a, beta1, beta2, sigma):
    t1 = 0.5*np.sqrt(sigma)*(a**2 + sigma**2)*(scipy.special.erf(beta2/(sigma*np.sqrt(2)))-scipy.special.erf(beta1/(sigma*np.sqrt(2))))
    if beta1 == -np.inf:
        t2 = 0
    else:
        t2 = sigma**(3/2) * (beta1-2*a)*np.exp(-beta1**2 / (2*sigma**2)) / np.sqrt(2*np.pi)
    if beta2 == np.inf:
        t3 = 0
    else:
        t3 = sigma**(3/2) * (beta2-2*a)*np.exp(-beta2**2 / (2*sigma**2)) / np.sqrt(2*np.pi)
    # print(t1, t2, t3)
    return t1+t2-t3
    
def lloyd_max_ent(M, lam, sigma):
    a = 3
    betas = np.linspace(-a, a, M-1)
    betas = np.insert(betas, 0, -np.inf)
    betas = np.append(betas, np.inf)
    ss = np.linspace(-a - 2*a/(M), a + 2*a/(M), M)
    
    ent = dist = 10
    ent_prev = dist_prev = 200
    while(abs(ent-ent_prev)+abs(dist-dist_prev) > 1e-5):
        pp = norm.cdf(sigma*betas[1:M+1])- norm.cdf(sigma*betas[0:M])
        ent_prev = ent
        dist_prev = dist
        ent = -np.inner(pp, np.log2(pp))
        dist = sum(np.array([dist_interval(ss[m], betas[m], betas[m+1], sigma) for m in range(M)]))
        # print(f'entropy={ent:.4f}, distortion={dist:.4f}')#, betas={betas}, ss={ss}')
        betas[1:M] = 0.5*(ss[1:M]+ss[0:M-1]) - lam*(np.log2(norm.cdf(sigma*betas[1:M])- norm.cdf(sigma*betas[0:M-1]))-np.log2(norm.cdf(sigma*betas[2:M+1])- norm.cdf(sigma*betas[1:M]))) / (2*sigma*(ss[0:M-1]-ss[1:M]))
        ss = sigma*(norm.pdf(betas[0:M]) - norm.pdf(betas[1:M+1])) / (norm.cdf(betas[1:M+1])- norm.cdf(betas[0:M]))
    return ent, dist, betas, ss

def lagrangian(M, lam, sigma, R):
    ent, dist, _, _ = lloyd_max_ent(M, lam, sigma)
    return dist + lam*(ent - R)

def find_quant(R, sigma):
    # M = np.ceil(2**R).astype('int')
    M = 5
    obj = lambda lam : -lagrangian(M, lam, sigma, R)
    lam_opt = scipy.optimize.minimize_scalar(obj, options={'disp':True}).x
    ent, dist, _, _ = lloyd_max_ent(M, lam_opt, sigma)
    return lam_opt, ent, dist
    

# RCC functions

def get_Ks(x_batch, Ys, lam_opt, sigma, Ws):
    return np.log(Ws) + (1/(2*lam_opt))*scipy.spatial.distance_matrix(x_batch[:,None], Ys[:,None])**2 
    # return np.log(Ws) + (1/(2*lam_opt))*scipy.spatial.distance_matrix(x_batch[:,None], Ys[:,None])**2 - (1/(2*(sigma**2+lam_opt)))*Ys[None, :]**2

def get_K(x_batch, Ys, lam_opt, sigmas, Ws):
    # return np.log(Ws) + (1/(2*lam_opt))*scipy.spatial.distance_matrix(x_batch[:,None], Ys[:,None])**2
    return scipy.spatial.distance_matrix(x_batch, Ys)**2 + 2*lam_opt * np.log(Ws)
    # return scipy.spatial.distance_matrix(x_batch, Ys)**2 - np.inner(lam_opt / (lam_opt + sigmas**2),Ys[None,:]**2) + 2*lam_opt * np.log(Ws)
    # return np.log(Ws) + (1/(2*lam_opt))*scipy.spatial.distance_matrix(x_batch, Ys)**2 - (1/(2*(sigma**2+lam_opt)))*Ys[None, :]**2

def compress_PFR_block_batch(x_batch, N, sigmas, lam_opt, ind_over, method='PFR'):
    Ys = np.sqrt(lam_opt + sigmas[None,:]**2)*np.random.randn(N, len(ind_over))
    if method == 'ORC':
        weights = np.array([N/(N-j) for j in range(N)])
    elif method == 'PFR':
        weights = np.ones(N)
    weighted_exp = weights*np.random.exponential(1, N)
    Ts = np.cumsum(weighted_exp)
    Ks = get_K(x_batch[:,ind_over], Ys, lam_opt, sigmas, Ts[None,:])
    # print(Ks.shape)
    K_batch = np.argmin(Ks, axis=1)
    return K_batch, Ys[K_batch]    
        
def compress_PFR_batch(x_batch, N, sigma, lam_opt, method='PFR'):
    Ys = np.sqrt(lam_opt + sigma**2)*np.random.randn(N)
    if method == 'ORC':
        weights = np.array([N/(N-j) for j in range(N)])
    elif method == 'PFR':
        weights = np.ones(N)
    weighted_exp = weights*np.random.exponential(1, N)
    Ts = np.cumsum(weighted_exp)
    Ks = get_Ks(x_batch, Ys, lam_opt, sigma, Ts[None,:])
    # print(Ks.shape)
    K_batch = np.argmin(Ks, axis=1)
    return K_batch, Ys[K_batch]

def vector_PFR(X, sigmas, D, N):
    lam_opt = rev_wf(sigmas, D)
    ind_over = np.where(sigmas**2 > lam_opt)[0]
    print(ind_over)
    K = np.zeros((X.shape[0], len(ind_over)))
    # print(K.shape)
    Y = np.zeros((X.shape[0], X.shape[1]))
    for k in ind_over:
        K_batch, Y_batch = compress_PFR_batch(X[:,k], N, sigmas[k], lam_opt)
        # print(K_batch.shape)
        K[:,k] = K_batch
        Y[:,k] = Y_batch
    rates = 0.5*np.log2(sigmas[ind_over]**2 / lam_opt)
    codebooks = [zipf_codebook(N, rate) for rate in rates]
    rates_PFR = [est_rate_zipf(K[:,k], codebooks[k]) for k in range(len(ind_over))]
    rate_PFR = sum(rates_PFR)
    dist_PFR = np.mean(np.linalg.norm(X-Y, axis=1)**2)
    return rate_PFR, dist_PFR

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

def est_rate_zipf_vector(K_mat, codebook):
    rate = np.zeros(K_mat.shape[1])
    for k in range(K_mat.shape[1]):
        rate[k] = est_rate_zipf(K_mat[:,k], codebook)
    return np.sum(rate)

def RD_PFR_block(D, sigmas, N, method='PFR'):
    X = np.random.multivariate_normal(np.zeros(sigmas.shape[0]), np.diag(sigmas**2), 1000)
    lam_opt = rev_wf(sigmas, D)
    ind_over = np.where(sigmas**2 > lam_opt)[0]
    codebook = zipf_codebook(N, rd_gaussian(D, sigmas))
    K, Y_partial = compress_PFR_block_batch(X, N, sigmas[ind_over], lam_opt, ind_over, method=method)
    Y = np.zeros(X.shape)
    Y[:,ind_over] = Y_partial
    r = est_rate_zipf(K, codebook)
    d = np.mean(np.linalg.norm(X-Y, axis=1)**2) 
    print(f'{D:.4f} finished')
    return r, d

def RD_PFR(D, sigmas, N):
    X = np.random.multivariate_normal(np.zeros(sigmas.shape[0]), np.diag(sigmas**2), 1000)
    r, d = vector_PFR(X, sigmas, D, N)
    return r, d