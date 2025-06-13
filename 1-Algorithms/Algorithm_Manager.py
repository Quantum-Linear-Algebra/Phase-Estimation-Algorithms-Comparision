import pickle
import sys
sys.path.append('.')
from Parameters import make_filename
sys.path.append('./1-Algorithms/Algorithms')
from ODMD import ODMD
from QCELS import QCELS
sys.path.append('./0-Data')

def run(parameters):
    filename = make_filename(parameters, add_shots=True)+".pkl"
    with open("0-Data/Expectation_Values/"+filename, 'rb') as file:
        exp_vals = pickle.load(file)

    print()
    for algo_name in parameters['algorithms']:
        run_single_algo(algo_name, exp_vals, filename, parameters)

def run_single_algo(algo_name, exp_vals, filename, parameters):
    print('Running', algo_name, 'with Dt =', parameters['Dt'])
    if algo_name == 'QCELS':
        est_E_0s, observables = QCELS(exp_vals, parameters['Dt'], skipping = 5)
    elif algo_name == 'ODMD':
        svd_threshold = 10**-3
        est_E_0s, observables = ODMD(exp_vals, parameters['Dt'], svd_threshold, parameters['num_timesteps'], show_steps=False)
    # readjust energy to what it originally was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']

    with open("1-Algorithms/Results/"+filename+'_'+str(algo_name), 'wb') as file:
        pickle.dump([observables, est_E_0s], file)
