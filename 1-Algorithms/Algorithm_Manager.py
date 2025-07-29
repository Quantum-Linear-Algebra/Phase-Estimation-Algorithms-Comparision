import pickle
import sys, os
sys.path.append('.')
from Parameters import make_filename, check_contains_linear
sys.path.append('./1-Algorithms/Algorithms')
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig
from ODMD import ODMD
from QCELS import QCELS
from VQPE import VQPE_ground_energy, UVQPE_ground_energy
from QMEGS import QMEGS_ground_energy
from ML_QCELS import ML_QCELS
sys.path.append('./0-Data')

def run(parameters, skipping=1):
    print('\nRunning Algorithms')

    all_exp_vals = {}
    for time_series in parameters['time_series']:
        (time_series_name, T, observables, shots, full_observable) = time_series
        with open('0-Data/Expectation_Values/'+make_filename(parameters, add_shots=True, shots=shots, key=time_series_name, T=T, obs=observables, fo = full_observable)+'.pkl', 'rb') as file:
            all_exp_vals[time_series] = pickle.load(file)
    
    for time_series in parameters['time_series']:
        (time_series_name, T, observables, shots, full_observable) = time_series
        algos = parameters['time_series'][time_series]
        for algo_name in algos:
            all_observables = []
            all_est_E_0s = []
            reruns = parameters['reruns']
            for series in all_exp_vals[time_series]:
                assert(len(series)>=reruns)
            for run in range(reruns):
                if algo_name != 'VQPE':
                    exp_vals = all_exp_vals[time_series][run]
                else:
                    Hev = all_exp_vals[time_series][run]
                    time_series_2 = ('linear', T, observables, shots, full_observable) # find some way of finding the right time series
                    if time_series_2 in all_exp_vals:
                        ev = all_exp_vals[time_series_2][run]
                    else:
                        for ts in all_exp_vals:
                            (ts_name2, T2, obs2, shots2, fo2) = ts
                            if ts_name2 == 'linear' and fo2 == full_observable and T/obs == T2/obs2 and shots==shots2:
                                ev = all_exp_vals[ts][run][:observables]
                    exp_vals = [ev, Hev]

                obs, est_E_0s = run_single_algo(algo_name, exp_vals, parameters, time_series, skipping=skipping)
                all_observables.append(obs)
                all_est_E_0s.append(est_E_0s)
            try: os.mkdir('1-Algorithms/Results')
            except: pass
            filename = make_filename(parameters, add_shots=True, T=T, obs=observables, shots=shots, fo=full_observable)+'.pkl'
            
            all_queries = [i*shots for i in all_observables[0]]
            with open('1-Algorithms/Results/'+algo_name+'_'+filename, 'wb') as file:
                pickle.dump([all_queries, all_est_E_0s], file)
            print('Saved', algo_name+'\'s results for T = ', T, ' into file.', '(1-Algorithms/Results/'+algo_name+'_'+filename+')')

def run_single_algo(algo_name, algo_exp_vals, parameters, time_series, skipping=1):
    (time_series_name, T, observables, shots, full_observable) = time_series
    if skipping > observables: skipping=observables
    if time_series_name == 'exp_vals' or time_series_name == 'sparse_exp_vals':
        algo_exp_vals[0] = 1 + 0j

    if algo_name == 'QCELS':
        Dt = T/observables
        est_E_0s, observables = QCELS(algo_exp_vals, Dt, parameters['algorithms']['QCELS']['lambda_prior'], skipping=skipping)
    elif algo_name == 'ODMD':
        Dt = T/observables
        threshold = parameters['algorithms']['ODMD']['svd_threshold']
        full_observable = parameters['algorithms']['ODMD']['full_observable']
        exp_vals = algo_exp_vals
        est_E_0s, observables = ODMD(exp_vals, Dt, threshold, len(exp_vals), full_observable=full_observable, skipping=skipping)
    elif algo_name == 'FDODMD':
        Dt = T/observables
        threshold = parameters['algorithms']['FDODMD']['svd_threshold']
        full_observable = parameters['algorithms']['FDODMD']['full_observable']
        fourier_params = {}
        gamma_range = parameters['algorithms']['FDODMD']['gamma_range']
        fourier_params['gamma_range'] = gamma_range
        filters = parameters['algorithms']['FDODMD']['filter_count']
        fourier_params['filters'] = filters
        exp_vals = algo_exp_vals
        est_E_0s, observables = ODMD(exp_vals, Dt, threshold, len(exp_vals), full_observable=full_observable, fourier_filter=True, fourier_params=fourier_params, skipping=skipping)
    elif algo_name == 'UVQPE':
        Dt = T/observables
        est_E_0s, observables = UVQPE_ground_energy(algo_exp_vals, Dt,  parameters['algorithms']['UVQPE']['svd_threshold'], skipping=skipping)
    elif algo_name == 'ML_QCELS':
        exp_vals = algo_exp_vals
        est_E_0s, observables = ML_QCELS(exp_vals, T, parameters['algorithms']['ML_QCELS']['time_steps'], parameters['algorithms']['ML_QCELS']['lambda_prior'])
    elif algo_name == 'VQPE':
        exp_vals = algo_exp_vals[0]
        Hexp_vals = algo_exp_vals[1]
        est_E_0s, observables = VQPE_ground_energy(exp_vals[:len(Hexp_vals)], Hexp_vals, len(parameters['algorithms']['VQPE']['pauli_strings']), parameters['algorithms']['VQPE']['svd_threshold'], skipping=skipping)
    elif algo_name == 'QMEGS':
        exp_vals = algo_exp_vals
        alpha = parameters['algorithms']['QMEGS']['alpha']
        q = parameters['algorithms']['QMEGS']['q']
        K = parameters['algorithms']['QMEGS']['K']
        full_observable = parameters['algorithms']['QMEGS']['full_observable']
        est_E_0s, observables = QMEGS_ground_energy(exp_vals, T, alpha, q, K, full_observable=full_observable, skipping=skipping)
    # readjust energy to what it originally was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']
    try:
        assert(len(observables)==len(est_E_0s))
        assert(len(est_E_0s)>0)
    except:
        print('observables:', observables)
        print('est_E_0s:', est_E_0s)
        print('There was a problem estimating the ground energy with', algo_name+'.')
        sys.exit(0)
    return observables, est_E_0s

# def gep_condition_number(A, B):
#     """
#     2-norm relative condition number of the simple eigenvalue lam.
#       (A,B)  : matrices
#       lam    : eigenvalue (scalar)
#       x, y   : right and left eigenvectors with y^H x = 1
#     """
#     # A, B given; compute eigen-pair first (e.g. with scipy.linalg.eig)
#     lam, X = eig(A, B)          # right eigenvectors (columns of X)
#     idx    = lam.argmin()       # pick an eigenvalue, say the largest
#     x      = X[:, idx]# compute corresponding left eigenvector
#     _, Y   = eig(A.T.conj(), B.T.conj())   # left eigenvectors
#     y      = Y[:, idx]
    
#     # make sure y^H x = 1
#     y = y / (y.conj().T @ x)
#     numerator   = norm(x) * norm(y)
#     denom       = abs(y.conj().T @ B @ x)
#     kappa_abs   = numerator / denom
#     kappa_rel   = kappa_abs * max(norm(A), norm(B)) / abs(lam)
#     print(f"abs κ = {kappa_abs:.2e},   rel κ = {kappa_rel:.2e}")
#     return kappa_abs, kappa_rel

if __name__ == '__main__':
    import sys 
    paths = ['.', './0-Data', './1-Algorithms']
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    from Comparison import parameters
    from Parameters import check
    check(parameters)
    run(parameters, skipping=1)