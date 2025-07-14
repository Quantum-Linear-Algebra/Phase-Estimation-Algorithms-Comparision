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
    # toeplitz().T sets the input to the top row instead of the first column
    H = toeplitz(Hexp_vals).T 
    if show_steps: print('H=',H)
    S = toeplitz(exp_vals).T
    if show_steps: print('S=',S)
    # eigh sorts eigenvalues
    d,V = eigh(S)
    d = d[::-1]
    V = V[:,::-1]
    filter = sum(abs(d)>svd_threshold*abs(d[0]))
    V = V[:,:filter]
    if show_steps: print('Eigenvalues',d)
    d = d[:filter]
    if show_steps: print('Filtered Eigenvalues',d)
    Ht = V.conj().T@H@V
    if show_steps: print('Ht=', Ht)
    St = np.diag(d)
    if show_steps: print('St=', St)
    eig_vals,_ = eig(Ht,St) # use eig instead of eigh because of hardware noise
    if show_steps: print('Eigenvalues:', eig_vals)
    return eig_vals.real

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
    if len(exp_vals)==2: col = exp_vals[1]
    else: col = np.concatenate([[exp_vals[1]], [exp_vals[0]],np.conj(exp_vals[1:-2])])
    H = toeplitz(col, exp_vals[1:])
    if show_steps: print('H=',H)
    # toeplitz().T sets the input to the top row instead of the first column
    S = toeplitz(exp_vals[:-1]).T
    if show_steps: print('S=',S)
    # eigh sorts eigenvalues
    d,V = eigh(S)
    d = d[::-1]
    V = V[:,::-1]
    if show_steps: print('d=',d)
    if show_steps: print('V=',V)
    filter = sum(abs(d)>svd_threshold*abs(d[0]))
    V = V[:,:filter]
    if show_steps: print('S Eigenvalues',d)
    d = d[:filter]
    if show_steps: print('Filtered S Eigenvalues',d)
    Ht = V.conj().T@H@V
    if show_steps: print('Ht=', Ht)
    St = np.diag(d)
    if show_steps: print('St=', St)   
    eig_vals,_ = eig(Ht,St)
    if show_steps: print('Exponentiated eigenvalues:', eig_vals)
    eig_vals = -(np.log(eig_vals)/Dt).imag
    if show_steps: print('Eigenvalues:', eig_vals)
    return eig_vals

def UVQPE_ground_energy(exp_vals, Dt,  svd_threshold, skipping=1, show_steps = False):
    est_E_0s = []
    observables = []
    for i in  range(len(exp_vals)):
        if i%skipping!=skipping-1: continue
        if i == 0: est_E_0s.append(0);continue
        if show_steps: print('\nIteration:', i+1)
        eig_vals = UVQPE(exp_vals[:i+1], Dt, svd_threshold, show_steps=show_steps)
        est_E_0s.append(eig_vals[0])
        observables.append((i+1)*2)
    return est_E_0s, observables

def VQPE_ground_energy(exp_vals, Hexp_vals, num_pauli_string, svd_threshold, skipping=1, show_steps = False):
    est_E_0s = []
    observables = []
    for i in  range(len(exp_vals)):
        if i%skipping!=skipping-1: continue
        if i == 0: est_E_0s.append(0); continue
        if show_steps: print('\nIteration:', i+1)
        eig_vals = VQPE(exp_vals[:i+1], Hexp_vals[:i+1], svd_threshold, show_steps=show_steps)
        est_E_0s.append(eig_vals[0])
        observables.append((num_pauli_string+1)*2*(i+1))
    return est_E_0s, observables
