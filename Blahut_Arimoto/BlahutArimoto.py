def run_BlahutArimoto(dist_mat, p_x, beta ,max_it = 500,eps = 1e-10) :
    """Compute the rate-distortion function of an i.i.d distribution
    Original author: Alon Kipnis
    Inputs :
        'dist_mat' -- (numpy matrix) representing the distoriton matrix between the input 
            alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j])
        'p_x' -- (1D numpy array) representing the probability mass function of the source
        'beta' -- (scalar) the slope of the rate-distoriton function at the point where evaluation is 
                    required
        'max_it' -- (int) maximal number of iterations
        'eps' -- (float) accuracy required by the algorithm: the algorithm stops if there
                is no change in distoriton value of more than 'eps' between consequtive iterations
    Returns :
        'Iu' -- rate (in bits)
        'Du' -- distortion
    """
    import numpy as np

    l,l_hat = dist_mat.shape
    p_cond = np.tile(p_x, (l_hat,1)).T #start with iid conditional distribution

    p_x = p_x / np.sum(p_x) #normalize
    p_cond /= np.sum(p_cond,1,keepdims=True)

    it = 0
    Du_prev = 0
    Du = 2*eps
    while it < max_it and np.abs(Du-Du_prev)> eps :
        
        it+=1
        Du_prev = Du
        p_hat = np.matmul(p_x,p_cond)

        p_cond = np.exp(-beta * dist_mat) * p_hat
        p_cond /= np.expand_dims(np.sum(p_cond,1),1)
        
        zeros = np.where(p_cond == 0)
        term = p_cond*np.log(p_cond / np.expand_dims(p_hat,0))
        term[zeros] = 0
        
        Iu = np.matmul(p_x, term).sum() / np.log(2)
        Du = np.matmul(p_x,(p_cond * dist_mat)).sum()
#         print(it, Iu, Du)
    return Iu, Du

