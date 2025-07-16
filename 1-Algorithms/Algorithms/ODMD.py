import numpy as np
from scipy.linalg import svd
from scipy.fft import fft, ifft

def print_matrix(mat):
    for i in mat:
        print("| ", end ="")
        for j in range(len(i)):
            if j != len(i)-1: print(f"{i[j]:10}", end="\t\t")
            else: print(i[j], end="")
        print(" |")

def make_hankel(s_ks):
    '''
    Create the hankel matrices

    Parameters:
     - s_ks: the quantum data sets

    Returns:
     - X: the hankel matrix
    '''
    
def make_hankel(s_ks):
    '''
    Create the hankel matrices

    Parameters:
     - s_ks: the quantum data sets

    Returns:
     - X: the hankel matrix
    '''
    rows, cols = s_ks.shape
    m = cols//2
    X = np.zeros((rows*m, cols+1-m), dtype=complex)
    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         X[i][j] = s_k[0][i+j]
    for i in range(cols+1-m):
        # print(s_ks[:,0])
        temp = s_ks[:,i:i+m].T
        X[:,i] = temp.ravel()
    return X


def check_convergence(data, precision):
    '''
    Check to see if the given dataset has converged

    Parameters:
     - data: the dataset to check convergence
     - percision: to what order to check percision
    
    Returns:
     - true if data has converged, false otherwise
    '''

    if len(data) <= 11: return False
    for i in range(10):
        print(data[-i]-data[-i-1])
        if data[-i]/precision-data[-i-1]/precision > precision: return False
    return True

def fourier_filter_exp_vals(exp_vals, gamma_range, filters):
    gammas = np.linspace(gamma_range[0], gamma_range[1], filters)
    filtered_exp_vals = []
    fft_exp_vals = fft(exp_vals)
    fft_median = np.median(np.abs(fft_exp_vals))
    # print(fft_median)
    for gamma in gammas:
        # true is counted as 1, false is counted as 0
        new_exp_vals = ifft([i*(abs(i)>gamma*fft_median) for i in fft_exp_vals]) 
        filtered_exp_vals.append(new_exp_vals)
    return filtered_exp_vals

def odmd(s_ks, Dt, svd_threshodl, precision=0, full_observable=True, fourier_filter=False, fourier_params={}):
    if len(s_ks[0]) == 0: return 0
    temp = make_hankel(s_ks) # [:,:k+1]
    return

def ODMD(s_k, Dt, svd_threshold, max_iterations, precision = 0, full_observable=True, fourier_filter=False, fourier_params={}, show_steps = False, skipping = 1):
    '''
    Preform the ODMD calculation.

    Parameters:
     - s_k: the quantum data
     - Dt: the time step
     - svd_threshold: effects the filtering of the system
                        matrix A. Should be one order higher
                        than noise level of backend
     - max_iterations: the maximum number of iterations to
                       do to estimate E_0 to precision
     - percision: the minimum precision desired
     - est_E_0s: a list to store the estimated E_0s
     - show_steps: if true then debugging print statements
                   are shown
    
    Returns:
     - E_0: the minimum ground energy estimate
    '''

    if not full_observable:
        s_k = [i.real for i in s_k]
    if not fourier_filter:
        s_ks = np.array([s_k])
    else:
        s_ks = np.array(fourier_filter_exp_vals(s_k, fourier_params['gamma_range'], fourier_params['filters']))
    # print(np.linalg.norm(s_k-s_ks[0]))
    k = -1
    est_E_0s = []
    observables = []
    while (True):
        k+=1
        if k%skipping!=skipping-1: continue
        if k>=max_iterations: break
        if show_steps: print("k =", k+1)
        if k < 1:
            E_0 = 0
        else:
            temp = make_hankel(s_ks[:,:k+1])
            X = temp[:,:-1]
            Xprime = temp[:,1:]
            if show_steps: print("X"); print_matrix(X)
            if show_steps: print("Xprime") ; print_matrix(Xprime)
            U, S, Vh = svd(X, full_matrices=False)
            r = np.sum(S > svd_threshold * S[0]) # Rank truncation
            U = U[:, :r]
            if show_steps: print("singular values:", S)
            S = S[:r]
            if show_steps: print("filtered singular values:", S)
            V = Vh[:r, :].conj().T
            S_inv = np.diag(1/S)
            A = U.conj().T @ Xprime @ V @ S_inv # atilde from ROEL_ODMD
            # print(np.linalg.matrix_rank(A))
            if show_steps: print("A"); print_matrix(A)  
            eigenvalues = np.linalg.eigvals(A)
            if show_steps: print("eigenvalues\n", eigenvalues)
            omega = np.sort(-np.imag(np.log(eigenvalues)/Dt))
            if show_steps: print("omega =", omega)
            E_0 = omega[0]
        obs = k+1
        if full_observable: obs *= 2
        observables.append(obs)
        est_E_0s.append(E_0)
        if show_steps: print("E_0 =", E_0)
        if precision!=0 and check_convergence(est_E_0s, precision): break
    return est_E_0s, observables


