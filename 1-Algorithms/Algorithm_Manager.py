import pickle
import sys
sys.path.append('.')
from Parameters import make_filename
sys.path.append('./1-Algorithms/Algorithms')
import numpy as np
from ODMD import ODMD
from QCELS import QCELS
from UVQPE import UVQPE_ground_energy
from ML_QCELS import ML_QCELS
sys.path.append('./0-Data')

def run(parameters, skipping=1):
    filename = make_filename(parameters, add_shots=True)+'.pkl'
    with open('0-Data/Expectation_Values/'+filename, 'rb') as file:
        exp_vals = pickle.load(file)

    print()
    for algo_name in parameters['algorithms']:
        run_single_algo(algo_name, exp_vals, filename, parameters, skipping=skipping)

def run_single_algo(algo_name, exp_vals, filename, parameters, skipping=1):
    print('Running', algo_name, 'with Dt =', parameters['Dt'])

    # Approximate what Hartree-Fock would estimate
    if algo_name == 'QCELS' or algo_name == 'ML_QCELS':
        order = np.floor(np.log10(-parameters['scaled_E_0']))
        lambda_prior = -(float(f"{parameters['scaled_E_0']:0.1}") + np.random.rand()*(10**(order)))
    
    if algo_name == 'QCELS':
        est_E_0s, observables = QCELS(exp_vals, parameters['Dt'], lambda_prior, skipping=skipping)
    elif algo_name == 'ODMD':
        svd_threshold = 10**-1
        est_E_0s, observables = ODMD(exp_vals, parameters['Dt'], svd_threshold, parameters['num_timesteps'], skipping=skipping)
    elif algo_name == 'UVQPE':
        svd_threshold = 10**-6
        est_E_0s, observables = UVQPE_ground_energy(exp_vals, parameters['Dt'], svd_threshold, skipping=skipping, show_steps=False)
    elif algo_name == 'ML_QCELS':
        time_steps = 5
        est_E_0s, observables = ML_QCELS(exp_vals, parameters['Dt'], time_steps, lambda_prior)
    # readjust energy to what it originally was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']
    with open('1-Algorithms/Results/'+algo_name+'_'+filename, 'wb') as file:
        pickle.dump([observables, est_E_0s], file)
    print('Saved', algo_name+'\'s results into file.', '(1-Algorithms/Results/'+algo_name+'_'+filename+')')

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