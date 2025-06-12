from scipy.linalg import eig
import numpy as np

def toeplitz(data):
    n = (len(data)-1)/2
    X = np.zeros((n, n), dtype=complex)
    X[0][0]=data[0]
    for x in range(1,n):
        X[x][0] = data[x]
        for y in range(1,n):
            X[0][y] = data[x+y]
            X[x][y]=X[x-1][y-1]
    return data

def VQPE(exp_vals, Hexp_vals, tol = 10**-6):
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
    S = toeplitz(exp_vals)
    V,d = eig(S)
    filter = sum(d>tol*d[0])
    V = V[:,:filter]
    d = d[:filter]
    Ht = V.conj().T*H*V
    Ht = (Ht + Ht.conj().T)/2
    St = np.diag(d)
    eig_vals,_ = eig(Ht,St)
    return eig_vals

def VQPE_ground_energy(exp_vals, Hexp_vals, tol = 0):
    results = []
    for i in range(len(exp_vals)):
        eig_vals = VQPE(exp_vals[:i], Hexp_vals[:i], tol=tol)
        results.append(eig_vals[0])
    return results
