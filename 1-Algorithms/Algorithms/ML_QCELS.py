import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

def arrange_Z_ests(old_Z_ests, ts, sparse):
    
    if sparse:
        max = np.sort(list(old_Z_ests.keys()))[-1]
        iterations = int(np.log2(max/ts))+1
    else:
        iterations = int(np.floor(np.log2((len(old_Z_ests) - 1)/(ts-1))) + 1)
    Z_ests = []
    for iter in range(iterations):
        Z_ests.append([])
        for t in range(ts):
            Z_ests[iter].append(old_Z_ests[(2**iter)*t])
    return Z_ests

def dt_to_index(smallest_dt, max_t):
    return max_t/smallest_dt

def closest_unitary(A):
    """ 
    Description: Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A. Used when qiskit fails to transpile
    unitary gate due to float point rounding.

    Args: Unitary matrix which qiskit fails to diagonalize: A

    Return: Unitary as an np matrix
    """
    V, __, Wh = svd(A)
    U = np.matrix(V.dot(Wh))
    return U

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex') # 'complex_'
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res

def get_tau(j, time_steps, epsilon, delta):
    return delta*(2**(j - 1 - np.ceil(np.log2(1/epsilon))))/(time_steps*(epsilon))

def qcels_largeoverlap(Z_est, time_steps, lambda_prior, tau):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: expectation values of time evolution: Z_est; 
    1/precision: T; 
    number of data pairs(time steps): time_steps; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0: res; 
    total time steps performed: t_ns; 
    """
    iterations = len(Z_est)
    ts=tau*np.arange(time_steps)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est[0], x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    for iter in range(iterations):
        ts=tau*np.arange(time_steps)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
        res = qcels_opt(ts, Z_est[iter], x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
        tau*=2
    return res

def ML_QCELS(Z_ests, Dt, ts, lambda_prior, sparse=False):
    observables = []
    est_E_0s = []
    Z_ests = arrange_Z_ests(Z_ests, ts, sparse=sparse)
    iterations = len(Z_ests)
    for iter in range(0, iterations):
        ground_energy_estimate_QCELS= qcels_largeoverlap(Z_ests[:iter+1], ts, lambda_prior, Dt)
        times = set()
        for itr in range(iter+1):
            time_series = Z_ests[itr]
            for time in time_series:
                times.add(time)
        observables.append(2*len(times))
        est_E_0 = ground_energy_estimate_QCELS.x[2] 
        est_E_0s.append(est_E_0)
    return est_E_0s, observables


