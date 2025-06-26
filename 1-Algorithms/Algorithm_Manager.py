import pickle
import sys, os
sys.path.append('.')
from Parameters import make_filename
sys.path.append('./1-Algorithms/Algorithms')
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig
from ODMD import ODMD
from QCELS import QCELS
from UVQPE import UVQPE_ground_energy, VQPE_ground_energy
from ML_QCELS import ML_QCELS
sys.path.append('./0-Data')

def run(parameters, skipping=1):
    filename = make_filename(parameters, add_shots=True)+'.pkl'
    all_exp_vals = []
    if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
        with open('0-Data/Expectation_Values/linear_'+filename, 'rb') as file:
            all_exp_vals = pickle.load(file)
    if parameters['const_obs'] and 'ML_QCELS' in parameters['algorithms']: 
        with open('0-Data/Expectation_Values/sparse_'+filename, 'rb') as file:
            all_sparse_exp_vals = pickle.load(file)
    if 'VQPE' in parameters['algorithms']:
        with open('0-Data/Expectation_Values/vqpets_'+filename, 'rb') as file:
            all_Hexp_vals = pickle.load(file)
    print()
    
    if 'QCELS' in parameters['algorithms'] or 'ML_QCELS' in parameters['algorithms']:
        # Approximate what Hartree-Fock would estimate
        E_0 = parameters['scaled_E_0']
        # lambda_prior = E_0
        order = np.floor(np.log10(np.abs(E_0)))
        digits = 2
        lambda_prior = -(int(str(E_0*10**(-order+digits))[1:digits+1])+np.random.rand())*(10**(order-digits+1))
        print('Lambda Prior for QCELS based methods:', lambda_prior)
    for algo_name in parameters['algorithms']:
        all_observables = []
        all_est_E_0s = []
        loop_num = len(all_exp_vals)
        if parameters['algorithms'] == ['ML_QCELS'] and parameters['const_obs']:
            loop_num = len(all_sparse_exp_vals)
        for run in range(loop_num):           
            if algo_name == 'ML_QCELS' and parameters['const_obs']: ev = all_sparse_exp_vals[run]
            else: ev = all_exp_vals[run]
            if algo_name == 'VQPE': Hexp_vals = all_Hexp_vals[run]

            print(str(run+1)+': Running', algo_name, 'with Dt =', parameters['Dt'])  
            if algo_name == 'QCELS' or algo_name == 'ML_QCELS': observables, est_E_0s = run_single_algo(algo_name, ev, parameters, skipping=skipping, lambda_prior=lambda_prior)
            elif algo_name == 'VQPE': observables, est_E_0s = run_single_algo(algo_name, ev, parameters, skipping=skipping, Hexp_vals=Hexp_vals)
            else: observables, est_E_0s = run_single_algo(algo_name, ev, parameters, skipping=skipping)
            all_observables.append(observables)
            all_est_E_0s.append(est_E_0s)
        try: os.mkdir('1-Algorithms/Results')
        except: pass
        with open('1-Algorithms/Results/'+algo_name+'_'+filename, 'wb') as file:
            pickle.dump([all_observables, all_est_E_0s], file)
        print('Saved', algo_name+'\'s results into file.', '(1-Algorithms/Results/'+algo_name+'_'+filename+')')

def run_single_algo(algo_name, exp_vals, parameters, skipping=1, lambda_prior=0, Hexp_vals=[]):  
    if algo_name == 'QCELS':
        assert(lambda_prior != 0)
        est_E_0s, observables = QCELS(exp_vals, parameters['Dt'], lambda_prior, skipping=skipping)
    elif algo_name == 'ODMD':
        est_E_0s, observables = ODMD(exp_vals, parameters['Dt'], parameters['ODMD_svd_threshold'], len(exp_vals), skipping=skipping)
    elif algo_name == 'UVQPE':
        est_E_0s, observables = UVQPE_ground_energy(exp_vals, parameters['Dt'],  parameters['UVQPE_svd_threshold'], skipping=skipping, show_steps = True)
    elif algo_name == 'ML_QCELS':
        assert(lambda_prior != 0)
        est_E_0s, observables = ML_QCELS(exp_vals, parameters['Dt'], parameters['ML_QCELS_time_steps'], lambda_prior, sparse=parameters['const_obs'])
    elif algo_name == 'VQPE':
        assert(len(Hexp_vals) != 0)
        est_E_0s, observables = VQPE_ground_energy(exp_vals[:len(Hexp_vals)], Hexp_vals, len(parameters['pauli_strings']), parameters['VQPE_svd_threshold'], skipping=skipping, show_steps=False)
    # readjust energy to what it originally was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']
    return observables, est_E_0s

def gep_condition_number(A, B):
    """
    2-norm relative condition number of the simple eigenvalue lam.
      (A,B)  : matrices
      lam    : eigenvalue (scalar)
      x, y   : right and left eigenvectors with y^H x = 1
    """
    # A, B given; compute eigen-pair first (e.g. with scipy.linalg.eig)
    lam, X = eig(A, B)          # right eigenvectors (columns of X)
    idx    = lam.argmin()       # pick an eigenvalue, say the largest
    x      = X[:, idx]# compute corresponding left eigenvector
    _, Y   = eig(A.T.conj(), B.T.conj())   # left eigenvectors
    y      = Y[:, idx]
    
    # make sure y^H x = 1
    y = y / (y.conj().T @ x)
    numerator   = norm(x) * norm(y)
    denom       = abs(y.conj().T @ B @ x)
    kappa_abs   = numerator / denom
    kappa_rel   = kappa_abs * max(norm(A), norm(B)) / abs(lam)

    
    
    print(f"abs κ = {k_abs:.2e},   rel κ = {k_rel:.2e}")
    return kappa_abs, kappa_rel

if __name__ == '__main__':
    import sys 
    paths = ['.', './0-Data', './1-Algorithms']
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    from Comparison import parameters
    from Parameters import check
    check(parameters)
    run(parameters)