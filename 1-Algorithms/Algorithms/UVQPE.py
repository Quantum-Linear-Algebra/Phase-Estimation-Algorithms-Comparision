from scipy.linalg import eig, toeplitz, eigh
import numpy as np

def VQPE(exp_vals, Hexp_vals, svd_threshold, show_steps = False):
    '''
    Estimates the energy states of the system
    represented by exp_vals and Hexp_vals 

    Parameters:
     - exp_vals: a series of expectation values of the
                 the time evolution operator
     - Hexp_vals: a series of the expectation values
                  of the time evolution operator times
                  the hamiltonian of the system
     - tol: the tolerance of the decomposition
    
    Returns:
     - eig_vals: the estimated eigenvalues of the system
    '''
    H = toeplitz(Hexp_vals)
    if show_steps: print('H=',H)
    S = toeplitz(exp_vals)
    if show_steps: print('S=',S)
    d,V = eig(S)
    idx = d.argsort()[::-1]
    d = d[idx]
    V = V[:,idx]
    filter = sum(abs(d)>svd_threshold*abs(d[0]))
    V = V[:,:filter]
    if show_steps: print('Singular Values',d)
    d = d[:filter]
    if show_steps: print('Filtered Singular Values',d)
    Ht = V.conj().T@H@V
    Ht = (Ht + Ht.conj().T)/2
    if show_steps: print('Ht=', Ht)
    St = np.diag(d)
    if show_steps: print('St=', St)
    eig_vals,_ = eig(Ht,St)
    # eig_vals = -(np.log(eig_vals)/Dt).imag
    if show_steps: print('Eigenvalues:', eig_vals)
    return eig_vals

def UVQPE(exp_vals, Dt, svd_threshold, show_steps = False):
    '''
    Estimates the energy states of the system
    represented by exp_vals and Hexp_vals 

    Parameters:
     - exp_vals: a series of expectation values of the
                 the time evolution operator
     - Dt: the time step for the series
     - svd_threshold: the tolerance of the svd decomposition
    
    Returns:
     - eig_vals: the estimated eigenvalues of the system
    '''
    if len(exp_vals)<=2: col = exp_vals[:len(exp_vals)]
    else: col = np.concatenate([[exp_vals[1]], [exp_vals[0]],np.conj(exp_vals[1:-2])])
    H = toeplitz(col, exp_vals[1:])
    if show_steps: print('H=',H)
    S = toeplitz(exp_vals[:-1])
    if show_steps: print('S=',S)
    d,V = eigh(S)
    idx = d.argsort()[::-1]
    d = d[idx]
    V = V[:,idx]
    filter = sum(abs(d)>svd_threshold*abs(d[0]))
    V = V[:,:filter]
    if show_steps: print('Singular Values',d)
    d = d[:filter]
    if show_steps: print('Filtered Singular Values',d)
    Ht = V.conj().T@H@V
    # Ht = (Ht + Ht.conj().T)/2
    if show_steps: print('Ht=', Ht)
    St = np.diag(d)
    if show_steps: print('St=', St)
    # from Algorithm_Manager import gep_condition_number
    # gep_condition_number(Ht, St, lam, x, y)
    

    eig_vals,_ = eig(Ht,St)
    if show_steps: print('Exponentiated eigenvalues:', eig_vals)
    eig_vals = -(np.log(eig_vals)/Dt).imag
    if show_steps: print('Eigenvalues:', eig_vals)
    return eig_vals

def UVQPE_ground_energy(exp_vals, Dt,  svd_threshold, skipping=1, show_steps = False):
    est_E_0s = []
    indexes = [i*skipping for i in  range(int(len(exp_vals)/skipping))]
    for i in indexes:
        if i < 2: est_E_0s.append(0);continue
        if show_steps: print('\nIteration:', i+1)
        eig_vals = UVQPE(exp_vals[:i+1], Dt, svd_threshold, show_steps=show_steps)
        est_E_0s.append(min(eig_vals))
    return est_E_0s, [2*i for i in indexes]

def VQPE_ground_energy(exp_vals, Hexp_vals, num_pauli_string, svd_threshold, skipping=1, show_steps = False):
    est_E_0s = []
    indices = [i*skipping for i in  range(int(len(Hexp_vals)/skipping))]
    for i in indices:
        if i < 2: est_E_0s.append(0); continue
        if show_steps: print('\nIteration:', i+1)
        eig_vals = VQPE(exp_vals[:i+1], Hexp_vals[:i+1], svd_threshold, show_steps=show_steps)
        est_E_0s.append(min(eig_vals))
    return est_E_0s, [(num_pauli_string+1)*2*(i+1) for i in indices]
