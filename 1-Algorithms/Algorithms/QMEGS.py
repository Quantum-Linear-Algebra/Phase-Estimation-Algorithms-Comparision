import numpy as np
from scipy.stats import truncnorm

def generate_ts_distribution(T,N,sigma):
    """ 
    Generate time samples from truncated Gaussian
    Input:
    T : variance of Gaussian
    sigma : truncated parameter
    N : number of samples
    
    Output: 
    t_list: np.array of time points
    """
    t_list=truncnorm.rvs(-sigma, sigma, loc=0, scale=T, size=N)
    return t_list

def QMEGS_algo(Z_est, d_x, t_list, alpha, T, K):
    """ 
    Main routines for QMEGS
    Goal: Given signal, output estimatation of dominant frequencies
    -Input:
    Z_est: np.array of signal
    d_x: space step
    t_list: np.array of time points
    K: number of dominant frequencies
    alpha: interval constant
    T: maximal time

    -Output:
    Dominant_freq: np.array of estimation of dominant frequencies (up to adjustment when there is no gap)
    """
    num_x=int(2*np.pi/(d_x*10))
    num_x_detail=int(2*alpha/d_x/T)
    x_rough=np.arange(0,num_x)*d_x*10-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x_rough)))/len(Z_est)) #Gaussian filter function
    Dominant_freq = []
    for k in range(K):
        max_idx_rough = np.argmax(G)
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*d_x+Dominant_potential-alpha/T
        G_detail=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est))
        max_idx_detail = np.argmax(G_detail)
        Dominant_freq.append(x[max_idx_detail])
        interval_max=x[max_idx_detail]+alpha/T
        interval_min=x[max_idx_detail]-alpha/T
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min)
    return np.sort(Dominant_freq)

def QMEGS_ground_energy(exp_vals,T_max,alpha,q, K, full_observable=True, skipping = 1):
    """ 
    Uses QMEGS to estimate ground state energy
    -Input:
    Z_ests: np.array of signal
    t_list: np.array of time points
    T_max: maximal time
    alpha: interval constant
    q: searching parameter

    -Output:
    output_energy: ground state energy estimate
    len(Z_ests): number of observables
    T_total_QMEGS: Total running time of QMEGS
    """
    t_list = list(exp_vals.keys())
    Z_ests = []
    for time in t_list:
        exp_val = exp_vals[time]

        re_p0 = (exp_val.real+1)/2
        if re_p0 > .5: Re = 1
        # elif re_p0 == .5: Re = 0
        else: Re = -1

        if full_observable:
            im_p0 = (exp_val.imag+1)/2
            if im_p0 > .5: Im = 1
            # elif im_p0 == .5: Im = 0
            else: Im = -1
        else: Im=0

        Z_ests.append(complex(Re, Im))
    Z_ests = np.array(Z_ests)
    t_list = np.array(t_list)
    # T_totals = []
    d_x=q/T_max
    output_energies = []
    observables = []
    for i in range(len(Z_ests)):
        if i%skipping!=skipping-1: continue
        # T_totals.append(sum(np.abs(t_list[:i])))
        Es = QMEGS_algo(Z_ests[:i+1], d_x, t_list[:i+1], alpha, T_max, K)
        output_energies.append(Es[0])
        obs = i+1
        if full_observable: obs *= 2
        observables.append(obs)
    return output_energies, observables#, T_totals