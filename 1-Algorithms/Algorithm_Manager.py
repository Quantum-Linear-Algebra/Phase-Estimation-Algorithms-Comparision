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

def run(parameters, skipping:int=1):
    print('\nRunning Algorithms')
    contains_linear = check_contains_linear(parameters['algorithms'], parameters['const_obs'])
    all_exp_vals = {}
    
    final_times = parameters['final_times']

    if contains_linear: all_exp_vals['linear'] = {}
    if parameters['const_obs'] and 'ML_QCELS' in parameters['algorithms']: all_exp_vals['sparse'] = {}
    if 'VQPE' in parameters['algorithms']: all_exp_vals['vqpets'] = {}
    if 'QMEGS' in parameters['algorithms']: all_exp_vals['gausts'] = {}

    for key in all_exp_vals:
        for T in final_times:
            with open('0-Data/Expectation_Values/'+make_filename(parameters, add_shots=True, key=key, T=T)+'.pkl', 'rb') as file:
                all_exp_vals[key][T] = pickle.load(file)
            if skipping> len(all_exp_vals[key][T][0]): skipping = len(all_exp_vals[key][T][0])
    for algo_name in parameters['algorithms']:
        for T in final_times:
            all_observables = []
            all_est_E_0s = []
            reruns = parameters['reruns']
            for key in all_exp_vals: assert(len(all_exp_vals[key][T])>=reruns)
            for run in range(reruns):
                algo_exp_vals = {}
                if parameters['const_obs'] and algo_name == 'ML_QCELS':
                    algo_exp_vals['sparse_exp_vals'] = all_exp_vals['sparse'][T][run]
                elif algo_name == 'QMEGS':
                    algo_exp_vals['gauss_exp_vals'] = all_exp_vals['gausts'][T][run]
                else: 
                    algo_exp_vals['exp_vals'] = all_exp_vals['linear'][T][run]
                    if algo_name == 'VQPE':
                        algo_exp_vals['Hexp_vals'] = all_exp_vals['vqpets'][T][run]
                observables, est_E_0s = run_single_algo(algo_name, algo_exp_vals, parameters, skipping=skipping)
                assert(len(observables)>0)
                assert(len(est_E_0s)>0)
                all_observables.append(observables)
                all_est_E_0s.append(est_E_0s)
            try: os.mkdir('1-Algorithms/Results')
            except: pass
            filename = make_filename(parameters, add_shots=True,T=T)+'.pkl'
            with open('1-Algorithms/Results/'+algo_name+'_'+filename, 'wb') as file:
                pickle.dump([all_observables, all_est_E_0s], file)
            print('Saved', algo_name+'\'s results for T = ', T, ' into file.', '(1-Algorithms/Results/'+algo_name+'_'+filename+')')

def run_single_algo(algo_name, algo_exp_vals, parameters, skipping=1):
    for key in algo_exp_vals:
        if key == 'exp_vals' or key == 'sparse_exp_vals':
            algo_exp_vals[key][0] = 1 + 0j

    if algo_name == 'QCELS':
        Dt = parameters['T']/parameters['observables']
        est_E_0s, observables = QCELS(algo_exp_vals['exp_vals'], Dt, parameters['QCELS_lambda_prior'], skipping=skipping)
    elif algo_name == 'ODMD':
        Dt = parameters['T']/parameters['observables']
        threshold = parameters['ODMD_svd_threshold']
        full_observable = parameters['ODMD_full_observable']
        exp_vals = algo_exp_vals['exp_vals']
        est_E_0s, observables = ODMD(exp_vals, Dt, threshold, len(exp_vals), full_observable=full_observable, skipping=skipping)
    elif algo_name == 'FODMD':
        Dt = parameters['T']/parameters['observables']
        threshold = parameters['FODMD_svd_threshold']
        full_observable = parameters['FODMD_full_observable']
        fourier_params = {}
        gamma_range = parameters['FODMD_gamma_range']
        fourier_params['gamma_range'] = gamma_range
        filters = parameters['FODMD_filter_count']
        fourier_params['filters'] = filters
        exp_vals = algo_exp_vals['exp_vals']
        est_E_0s, observables = ODMD(exp_vals, Dt, threshold, len(exp_vals), full_observable=full_observable, fourier_filter=True, fourier_params=fourier_params, skipping=skipping)
    elif algo_name == 'UVQPE':
        Dt = parameters['T']/parameters['observables']
        est_E_0s, observables = UVQPE_ground_energy(algo_exp_vals['exp_vals'], Dt,  parameters['UVQPE_svd_threshold'], skipping=skipping)
    elif algo_name == 'ML_QCELS':
        Dt = parameters['T']/parameters['observables']
        sparse = parameters['const_obs']
        if sparse: exp_vals = algo_exp_vals['sparse_exp_vals']
        else: exp_vals = algo_exp_vals['exp_vals']
        est_E_0s, observables = ML_QCELS(exp_vals, Dt, parameters['ML_QCELS_time_steps'], parameters['QCELS_lambda_prior'], sparse=sparse)
    elif algo_name == 'VQPE':
        exp_vals = algo_exp_vals['exp_vals']
        Hexp_vals = algo_exp_vals['Hexp_vals']
        est_E_0s, observables = VQPE_ground_energy(exp_vals[:len(Hexp_vals)], Hexp_vals, len(parameters['pauli_strings']), parameters['VQPE_svd_threshold'], skipping=skipping)
    elif algo_name == 'QMEGS':
        exp_vals = algo_exp_vals['gauss_exp_vals']
        T = parameters['T']
        alpha = parameters['QMEGS_alpha']
        q = parameters['QMEGS_q']
        K = parameters['QMEGS_K']
        est_E_0s, observables = QMEGS_ground_energy(exp_vals, T, alpha, q, K, skipping=skipping)
    # readjust energy to what it originally was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']
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
    run(parameters, skipping=int(parameters['observables']/2))