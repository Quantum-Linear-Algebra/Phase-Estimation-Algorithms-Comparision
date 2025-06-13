import numpy as np
from scipy.linalg import svd

def print_matrix(mat):
    for i in mat:
        print("| ", end ="")
        for j in range(len(i)):
            if j != len(i)-1: print(f"{i[j]:10}", end="\t\t")
            else: print(i[j], end="")
        print(" |")

def make_hankel(k, ref, s_k):
    '''
    Create the hankel matrices

    Parameters:
     - k: the amount of data to use
     - ref: the number of reference states (usually 1)
     - s_k: the quantum data

    Returns:
     - X: the hankel matrix
    '''
    m = k//2
    X = np.zeros((ref*m, k+1-m), dtype=complex)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = s_k[i+j]
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

def ODMD(s_k, Dt, svd_threshold, max_iterations, precision = 0, show_steps = False, skipping = 1):
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
    if show_steps: print("s_k:", s_k)
    k = -skipping
    est_E_0s = []
    while (True):
        k += skipping
        if k>=max_iterations: break
        if show_steps: print("k =", k+1)
        if k < 3: est_E_0s.append(0); continue # svd breaks if k<3
        temp = make_hankel(k, 1, s_k)
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
        est_E_0s.append(E_0)
        if show_steps: print("E_0 =", E_0)
        if precision!=0 and check_convergence(est_E_0s, precision): break
    return est_E_0s, [(i*skipping + 1)*2 for i in range(len(s_k)//skipping)]


