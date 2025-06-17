from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from numpy import pi
from scipy.linalg import eigh
from Service import create_hardware_backend
from sys import exit

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
        assert(algo in ['UVQPE','ODMD','QCELS'])

    if parameters['comp_type'] != 'J':
        variables = ['comp_type', 'algorithms', 'sites', 'Dt', 'scaling', 'shifting', 'overlap', 'system', 'num_timesteps', 'r_scaling']
        if parameters['comp_type'] != 'C': variables.append('shots')
        # verify system parameters are setup correctly
        if parameters['system'] == 'TFI':
            variables.append('g')
            if parameters['comp_type'] != 'C':
                variables.append('method_for_model')
                parameters['method_for_model'] = parameters['method_for_model'][0].upper()
                assert(parameters['method_for_model']=='F' or parameters['method_for_model']=='Q')
                if parameters['method_for_model'] == 'F': variables.append('trotter')
        elif parameters['system'] == 'HUB':
            variables.append('t')
            variables.append('U')
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
            variables.append('x')
            variables.append('y')
        elif parameters['system'] == 'SPI':
            variables.append('J')
            assert(parameters['J']!=0)
        elif parameters['system'] == 'H_2':
            variables.append('distance')
            parameters['sites']=1
        import sys
        sys.path.append('0-Data')
        from Data_Manager import create_hamiltonian
        H,_ =create_hamiltonian(parameters)
        energy,_ = eigh(H)
        print('Scaled Ground energy:', energy[0])
    else:
        variables = ['comp_type', 'algorithms']
    
    keys = []
    for i in parameters.keys():
        keys.append(i)
    for key in keys:
        if key not in variables:
            parameters.pop(key)

    # backend setup
    if parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        backend = create_hardware_backend()
    else:
        backend = AerSimulator(noise_model = NoiseModel())
    
    
    print('Parameters are setup:')
    for key in parameters.keys():
        print('  '+key+':', parameters[key])
    print()
    return backend


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
    string += '_maxitr='+str(parameters['num_timesteps'])
    if add_shots and parameters['comp_type'] != 'C': string += '_shots='+str(parameters['shots'])
    return string

