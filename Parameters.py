from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.linalg import eigh
from numpy import ceil, sqrt, zeros, log10, floor, abs, random, linspace
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
    if 'distribution' in parameters: assert(0.9999999999999999<=sum(parameters['distribution'])<=1.0000000000000099) # rounding
    for algo in parameters['algorithms']:
        assert(algo in ['VQPE','UVQPE','ODMD','FODMD','QCELS','ML_QCELS','QMEGS'])

    
    # verify system parameters are setup correctly
    returns = {}
    used_variables = []
    if parameters['comp_type'] == 'J':
        batch_id = input('Enter Job/Batch ID: ')
        print('Loading parameter data.')
        algos = parameters['algorithms']
        with open('0-Data/Jobs/'+str(batch_id)+'.pkl', 'rb') as file:
            [params, job_ids] = pickle.load(file)
        for key in params:
            used_variables.append(key)
            parameters[key] = params[key]
        parameters['algorithms'] = algos
        parameters['comp_type'] = 'J'
        returns['job_ids'] = job_ids
    else:
        used_variables = ['comp_type', 'algorithms', 'sites', 'T', 'scaling', 'shifting', 'system', 'observables', 'r_scaling', 'const_obs', 'real_E_0', 'scaled_E_0', 'reruns', 'sv', 'shots']
        parameters['T'] = float(parameters['T'])
        assert(parameters['T']>0)
        if parameters['comp_type'] == 'C' or 'shots' not in parameters: parameters['shots'] = 1
        if parameters['comp_type'] == 'C' or 'reruns' not in parameters: parameters['reruns'] = 1
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
            used_variables.append('overlap')
            parameters['sv'] = make_overlap(eig_vec[:,0], parameters['overlap'])
        elif 'distribution' in parameters:
            used_variables.append('distribution')
            parameters['sv'] = zeros(len(eig_vec[:,0]), dtype=complex)
            for i in range(len(parameters['distribution'])):
                # print(i, parameters['distribution'])
                parameters['sv'] += sqrt(parameters['distribution'][i])*eig_vec[:,i]
                # print(parameters['sv']@eig_vec[:,i])
            # assert(parameters['sv']@eig_vec[:,0]==parameters['distribution'][0]) 
        else: parameters['sv'] = eig_vec[:,0]
        parameters['scaled_E_0'] = energy[0]
        
        if 'const_obs' not in parameters: parameters['const_obs'] = False
        used_variables.append('final_times')
        if not parameters['const_obs']:
            num_sims = 10
            if 'num_time_sims' in parameters: num_sims = parameters['num_time_sims']
            parameters['final_times'] = linspace(0, parameters['T'], num_sims+1)[1:] # excluding 0
        else:
            parameters['final_times'] = [parameters['T']]

    if 'VQPE' in parameters['algorithms']:
        used_variables.append('VQPE_svd_threshold')
        if 'VQPE_svd_threshold' not in parameters: parameters['VQPE_svd_threshold'] = 10**-6
        used_variables.append('pauli_strings')
        parameters['pauli_strings'] = SparsePauliOp.from_operator(Operator(H))
        total_num_time_series = 2*(len(parameters['pauli_strings'])+1)
        if parameters['const_obs'] and parameters['observables']%total_num_time_series!=0:
            parameters['observables'] = int(ceil(parameters['observables']/total_num_time_series)*total_num_time_series)
    if 'QCELS' in parameters['algorithms'] or 'ML_QCELS' in parameters['algorithms']:
        # Approximate what Hartree-Fock would estimate
        if 'QCELS_lambda_prior' in parameters:
            lambda_prior = parameters['QCELS_lambda_prior']
        else:
            E_0 = parameters['scaled_E_0']
            order = floor(log10(abs(E_0)))
            if 'QCELS_lambda_digits' in parameters:
                digits = parameters['QCELS_lambda_digits']
                if digits == -1: digits = int(random.randint(1,3))
            else: digits = 2
            used_variables.append('QCELS_lambda_prior')
            lambda_prior = -(int(str(E_0*10**(-order+digits))[1:digits+1])+random.rand())*(10**(order-digits+1))
        parameters['QCELS_lambda_prior'] = lambda_prior
    if 'ML_QCELS' in parameters['algorithms']:
        # make sure the time steps per iteration is defined
        used_variables.append('ML_QCELS_time_steps')
        if 'ML_QCELS_time_steps' not in parameters: parameters['ML_QCELS_time_steps'] = 5
        # adjust the observables so that all algorithms match ML_QCELS's observables
        iteration = 0
        time_steps_per_itr = parameters['ML_QCELS_time_steps']
        times = set()
        while len(times) < parameters['observables']/2:
            for i in range(time_steps_per_itr):
                times.add(2**iteration*i)
            iteration+=1
        parameters['observables'] = len(times)*2
        if 'ML_QCELS_calc_Dt' in parameters and parameters['ML_QCELS_calc_Dt']:
            delta = 1*sqrt(1-parameters['overlap'])
            parameters['T'] = parameters['observables']*delta/parameters['ML_QCELS_time_steps']
    if 'ODMD' in parameters['algorithms']:
        used_variables.append('ODMD_svd_threshold')
        if 'ODMD_svd_threshold' not in parameters: parameters['ODMD_svd_threshold'] = 10**-6
        used_variables.append('ODMD_full_observable')
        if 'ODMD_full_observable' not in parameters: parameters['ODMD_full_observable'] = False
    if 'FODMD' in parameters['algorithms']:
        used_variables.append('FODMD_svd_threshold')
        if 'FODMD_svd_threshold' not in parameters: parameters['FODMD_svd_threshold'] = 10**-6
        used_variables.append('FODMD_full_observable')
        if 'FODMD_full_observable' not in parameters: parameters['FODMD_full_observable'] = False
        used_variables.append('FODMD_gamma_range')
        if 'FODMD_gamma_range' not in parameters:
            parameters['FODMD_gamma_range'] = (1,3)
        else:
            assert(parameters['FODMD_gamma_range'][0]>=0 and parameters['FODMD_gamma_range'][1]>=0)
            assert(parameters['FODMD_gamma_range'][0]<=parameters['FODMD_gamma_range'][1])
        used_variables.append('FODMD_filter_count')
        if 'FODMD_filter_count' not in parameters: parameters['FODMD_filter_count'] = 6
    if 'UVQPE' in parameters['algorithms']:
        used_variables.append('UVQPE_svd_threshold')
        if 'UVQPE_svd_threshold' not in parameters: parameters['UVQPE_svd_threshold'] = 10**-6
    if 'QMEGS' in parameters['algorithms']:
        used_variables.append('QMEGS_sigma')
        if 'QMEGS_sigma' not in parameters: parameters['QMEGS_sigma'] = 0.5
        used_variables.append('QMEGS_q')
        if 'QMEGS_q' not in parameters: parameters['QMEGS_q'] = 0.05
        used_variables.append('QMEGS_alpha')
        if 'QMEGS_alpha' not in parameters: parameters['QMEGS_alpha'] = 5
        used_variables.append('QMEGS_K')
        if 'QMEGS_K' not in parameters: parameters['QMEGS_K'] = 1
        
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
def make_filename(parameters, add_shots = False, key='', T = -1):
    system = parameters['system']
    
    string = ''
    if key != '': string += key+'_'
    string += 'comp='+parameters['comp_type']+'_sys='+system
    string +='_n='+str(parameters['sites'])
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
    if 'distribution' in parameters:
        string+='_distr=['
        for i in parameters['distribution'][:3][:-1]:
            string+=f'{i:0.2},'
        var = parameters['distribution'][3]
        string+=f'{var:0.2}]'
    if T == -1: string+='_T='+str(parameters['T'])
    else: string+='_T='+str(T)
    if parameters['algorithms'] == ['VQPE'] and parameters['const_obs']:
        string += '_obs='+str(int(parameters['observables']/(len(parameters['pauli_strings'])+1)))
    else:
        string += '_obs='+str(parameters['observables'])
    if key == 'gausts':
        string += '_sigma='+str(parameters['QMEGS_sigma'])
    if add_shots:
        string += '_reruns='+str(parameters['reruns'])
        if parameters['comp_type'] != 'C':
            string += '_shots='+str(parameters['shots'])
    return string

def check_contains_linear(algos):
    linear = ['ODMD', 'FODMD', 'VQPE', 'UVQPE', 'QCELS']
    for algo in algos:
        if algo in linear:
            return True
    return False
    

if __name__ == '__main__':
    from Comparison import parameters
    check(parameters)

