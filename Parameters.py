from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.linalg import eigh
from numpy import ceil, sqrt, zeros
from Service import create_hardware_backend
from sys import exit
import pickle
from qiskit.quantum_info import Operator, SparsePauliOp

def check(parameters):
    print('Setting up parameters.')
    
    # PREPROCESSING
    parameters['comp_type'] = parameters['comp_type'][0].upper()
    parameters['system']    = parameters['system'][0:3].upper()

    # parameter checking (if there's an error change parameters in question)
    assert(parameters['comp_type'] == 'C' or parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J')
    assert(parameters['system'] == 'TFI' or parameters['system'] == 'SPI' or parameters['system'] == 'HUB' or parameters['system'] == 'H_2')
    if 'overlap' in parameters: assert(0<=parameters['overlap']<=1)
    if 'distribution' in parameters: assert(sum(parameters['distribution'])==1)
    for algo in parameters['algorithms']:
        assert(algo in ['VQPE','UVQPE','ODMD','QCELS','ML_QCELS'])

    
    returns = {}
    # verify system parameters are setup correctly
    if parameters['comp_type'] == 'J':
        batch_id = input('Enter Job/Batch ID: ')
        print('Loading parameter data.')
        algos = parameters['algorithms']
        with open('0-Data/Jobs/'+str(batch_id)+'.pkl', 'rb') as file:
            [params, job_ids] = pickle.load(file)
        for key in list(parameters.keys()):
            parameters.pop(key)
        for key in params:
            parameters[key] = params[key]
        parameters['algorithms'] = algos
        parameters['comp_type'] = 'J'
        returns['job_ids'] = job_ids
    else:
        used_variables = ['comp_type', 'algorithms', 'sites', 'Dt', 'scaling', 'shifting', 'overlap', 'system', 'observables', 'r_scaling', 'const_obs', 'real_E_0', 'scaled_E_0', 'reruns', 'sv']
        if parameters['comp_type'] != 'C':
            used_variables.append('shots')
            if 'shots' not in parameters: parameters['shots'] = 100
            if 'reruns' not in parameters: parameters['reruns'] = 1
        else:
            parameters['reruns'] = 1
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
        
        import sys
        sys.path.append('0-Data')
        from Data_Manager import create_hamiltonian, make_overlap
        H,real_E_0 =create_hamiltonian(parameters)
        parameters['real_E_0'] = real_E_0
        energy,eig_vec = eigh(H)
        if 'overlap' in parameters:
            parameters['sv'] = make_overlap(eig_vec[:,0], parameters['overlap'])
        elif 'distribution' in parameters:
            parameters['sv'] = zeros(len(eig_vec[:,0]), dtype=complex)
            for i in range(len(parameters['distribution'])):
                print(i, parameters['distribution'])
                parameters['sv'] += sqrt(parameters['distribution'][i])*eig_vec[:,i]
                print(parameters['sv']@eig_vec[:,i])
            # assert(parameters['sv']@eig_vec[:,0]==parameters['distribution'][0]) 
        else: parameters['sv'] = eig_vec[:,0]
        parameters['scaled_E_0'] = energy[0]
        
        if 'const_obs' not in parameters: parameters['const_obs'] = False

        if 'VQPE' in parameters['algorithms']:
            used_variables.append('VQPE_svd_threshold')
            if 'VQPE_svd_threshold' not in parameters: parameters['VQPE_svd_threshold'] = 10**-6
            used_variables.append('pauli_strings')
            parameters['pauli_strings'] = SparsePauliOp.from_operator(Operator(H))
            total_num_time_series = 2*(len(parameters['pauli_strings'])+1)
            if parameters['const_obs'] and parameters['observables']%total_num_time_series!=0:
                parameters['observables'] = int(ceil(parameters['observables']/total_num_time_series)*total_num_time_series)
        if 'ML_QCELS' in parameters['algorithms']:
            # make sure the time steps per iteration is defined
            used_variables.append('ML_QCELS_time_steps')
            if 'ML_QCELS_time_steps' not in parameters: parameters['ML_QCELS_time_steps'] = 5
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
                delta = 1*sqrt(1-parameters['overlap'])
                parameters['Dt'] = delta/parameters['ML_QCELS_time_steps']
        if 'ODMD' in parameters['algorithms']:
            used_variables.append('ODMD_svd_threshold')
            if 'ODMD_svd_threshold' not in parameters: parameters['ODMD_svd_threshold'] = 10**-6
            used_variables.append('ODMD_full_observable')
            if 'ODMD_full_observable' not in parameters: parameters['ODMD_full_observable'] = False
        if 'UVQPE' in parameters['algorithms']:
            used_variables.append('UVQPE_svd_threshold')
            if 'UVQPE_svd_threshold' not in parameters: parameters['UVQPE_svd_threshold'] = 10**-6
        
        used_variables.append('fourier_filtering')
        if 'fourier_filtering' in parameters:
            if parameters['fourier_filtering']:
                used_variables.append('gamma_range')
                if 'gamma_range' not in parameters:
                    parameters['gamma_range'] = (1,3)
                else:
                    assert(parameters['gamma_range'][0]<parameters['gamma_range'][1])
                used_variables.append('filter_count')
                if 'filter_count' not in parameters:
                    parameters['filter_count'] = 6
                else:
                    assert(parameters['filter_count']>0)
        else:
            parameters['fourier_filtering'] = False
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
def make_filename(parameters, fourier_filtered=False, add_shots = False):
    system = parameters['system']
    string = 'comp='+parameters['comp_type']+'_sys='+system
    string+='_n='+str(parameters['sites'])
    if fourier_filtered:
        string+='_gamma='+str(parameters['gamma_range'][0])+','+str(parameters['gamma_range'][1])
        string+='_filters='+str(parameters['filter_count'])
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
    if 'overlap' in parameters: string+='_overlap='+str(parameters['overlap'])
    if 'distribution' in parameters: string+='_distr='+str(parameters['distribution'])
    string+='_Dt='+str(parameters['Dt'])
    if parameters['algorithms'] == ['VQPE'] and parameters['const_obs']:
        string += '_obs='+str(int(parameters['observables']/(len(parameters['pauli_strings'])+1)))
    else:
        string += '_obs='+str(parameters['observables'])
    if add_shots and parameters['comp_type'] != 'C':
        string += '_reruns='+str(parameters['reruns'])
        string += '_shots='+str(parameters['shots'])
    return string

