from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from numpy import pi, sqrt
from scipy.linalg import eigh
from Service import create_hardware_backend
from sys import exit
import pickle

def check(parameters):
    print('Setting up parameters.')
    
    # PREPROCESSING
    parameters['comp_type'] = parameters['comp_type'][0].upper()
    parameters['system']    = parameters['system'][0:3].upper()

    # parameter checking (if there's an error change parameters in question)
    assert(parameters['comp_type'] == 'C' or parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J')
    assert(parameters['system'] == 'TFI' or parameters['system'] == 'SPI' or parameters['system'] == 'HUB' or parameters['system'] == 'H_2')
    if 'overlap' not in parameters: parameters['overlap'] = 1
    assert(0<=parameters['overlap']<=1)
    for algo in parameters['algorithms']:
        assert(algo in ['UVQPE','ODMD','QCELS', 'ML_QCELS'])

    
    returns = {}
    # verify system parameters are setup correctly
    if parameters['comp_type'] == 'J':
        batch_id = input('Enter Job/Batch ID: ')
        print('Loading parameter data.')
        algos = parameters['algorithms']
        with open('0-Data/Jobs/'+str(batch_id)+'.pkl', 'rb') as file:
            [parameters, job_ids] = pickle.load(file)
        parameters['algorithms'] = algos
        returns['job_ids'] = job_ids
    else:
        used_variables = ['comp_type', 'algorithms', 'sites', 'Dt', 'scaling', 'shifting', 'overlap', 'system', 'observables', 'r_scaling', 'const_obs', 'real_E_0', 'scaled_E_0']
        if parameters['comp_type'] != 'C': used_variables.append('shots')
        if parameters['system'] == 'TFI':
            used_variables.append('g')
            if parameters['comp_type'] != 'C':
                used_variables.append('method_for_model')
                parameters['method_for_model'] = parameters['method_for_model'][0].upper()
                assert(parameters['method_for_model']=='F' or parameters['method_for_model']=='Q')
                if parameters['method_for_model'] == 'F': used_variables.append('trotter')
        elif parameters['system'] == 'HUB':
            used_variables.append('t')
            used_variables.append('U')
            x_in = 'x' in parameters.keys()
            y_in = 'y' in parameters.keys()
            if not x_in and not y_in:
                parameters['x'] = parameters['sites']
                parameters['y'] = 1
            elif not x_in: parameters['x'] = 1
            elif not y_in: parameters['y'] = 1
            x = parameters['x']
            y = parameters['y']
            assert(x>=0 and y>=0)
            assert(x*y == parameters['sites']) # change the latice shape
            used_variables.append('x')
            used_variables.append('y')
        elif parameters['system'] == 'SPI':
            used_variables.append('J')
            assert(parameters['J']!=0)
        elif parameters['system'] == 'H_2':
            used_variables.append('distance')
            parameters['sites']=1
        # remove extra keys
        import sys
        sys.path.append('0-Data')
        from Data_Manager import create_hamiltonian
        H,real_E_0 =create_hamiltonian(parameters)
        parameters['real_E_0'] = real_E_0
        energy,_ = eigh(H)
        parameters['scaled_E_0'] = energy[0]
        
        if 'const_obs' not in parameters: parameters['const_obs'] = False
        
        if 'ML_QCELS' in parameters['algorithms']:
            # make sure the time steps per iteration is defined
            if 'ML_QCELS_time_steps' not in parameters: parameters['ML_QCELS_time_steps'] = 5
            used_variables.append('ML_QCELS_time_steps')
            # adjust the observables so that all algorithms match ML_QCELS's observables
            if parameters['const_obs']:
                iteration = 0
                time_steps_per_itr = parameters['ML_QCELS_time_steps']
                exp_vals = set()
                while len(exp_vals) < parameters['observables']/2:
                    for i in range(time_steps_per_itr):
                        time = 2**iteration*i
                        if time in exp_vals: continue
                        exp_vals.add(time)
                    iteration+=1
                parameters['observables'] = len(exp_vals)*2
            if 'ML_QCELS_calc_Dt' in parameters and parameters['ML_QCELS_calc_Dt']:
                delta = 1 #*sqrt(1-parameters['overlap'])
                parameters['Dt'] = delta/parameters['ML_QCELS_time_steps']
        if 'ODMD' in parameters['algorithms']:
            if 'ODMD_svd_threshold' not in parameters: parameters['ODMD_svd_threshold'] = 10**-6
            used_variables.append('ODMD_svd_threshold')
        if 'UVQPE' in parameters['algorithms']:
            if 'UVQPE_svd_threshold' not in parameters: parameters['UVQPE_svd_threshold'] = 10**-6
            used_variables.append('UVQPE_svd_threshold')
        
        keys = []
        for i in parameters.keys():
            keys.append(i)
        for key in keys:
            if key not in used_variables:
                parameters.pop(key)

    # backend setup
    if parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        backend = create_hardware_backend()
    else:
        backend = AerSimulator(noise_model = NoiseModel())
    returns['backend'] = backend
    print('Parameters are setup:')
    for key in parameters.keys():
        print('  '+key+':', parameters[key])
    print()
    return returns


# define a system for naming files
def make_filename(parameters, add_shots = False):
    system = parameters['system']
    string = 'comp='+parameters['comp_type']+'_sys='+system
    string+='_n='+str(parameters['sites'])
    if system=='TFI':
        if parameters['comp_type'] != 'C':
            method_for_model = parameters['method_for_model']
            string+='_m='+method_for_model
            if method_for_model == 'F':
                string+='_trotter='+str(parameters['trotter'])
        string+='_g='+str(parameters['g'])
    elif system=='SPI':
        string+='_J='+str(parameters['J'])
    elif system=='HUB':
        string+='_t='+str(parameters['t'])
        string+='_U='+str(parameters['U'])
        string+='_x='+str(parameters['x'])
        string+='_y='+str(parameters['y'])
    elif system=='H_2':
        string+='_dist='+str(parameters['distance'])
    string+='_scale='+str(parameters['scaling'])
    string+='_shift='+str(parameters['shifting'])
    string+='_overlap='+str(parameters['overlap'])
    string+='_Dt='+str(parameters['Dt'])
    string += '_obs='+str(parameters['observables'])
    if add_shots and parameters['comp_type'] != 'C': string += '_shots='+str(parameters['shots'])
    return string

