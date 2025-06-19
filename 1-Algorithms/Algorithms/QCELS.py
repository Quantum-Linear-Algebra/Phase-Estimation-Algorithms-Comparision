import numpy as np
from scipy.optimize import minimize

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

def base_qcels_largeoverlap(exp_vals, lambda_prior, Dt):
    """Base-level QCELS for a system with a large initial overlap.

    Description: The code of using Base-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: expectation values of time evolution: exp_vals; 
    initial guess of \lambda_0: lambda_prior;

    Returns: a list of \lambda_0 estimates: est_E_0s; 
    """
    ts=Dt*np.arange(len(exp_vals))
    # Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, exp_vals, x0)# Solve the optimization problem
    return res.x[2]

def QCELS(exp_vals, Dt, lambda_prior, skipping = 1):
    time_steps = len(exp_vals)
    est_E_0s = []
    for i in range(time_steps//skipping):
        #------------------QCELS-----------------
        time_step_idx = i*skipping
        est_E_0 = base_qcels_largeoverlap(exp_vals[:time_step_idx], lambda_prior, Dt)
        est_E_0s.append(est_E_0)

    return est_E_0s, [(i*skipping + 1)*2 for i in range(time_steps//skipping)]

