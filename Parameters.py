from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService as QRS
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from numpy import pi
from Service import create_hardware_backend

print("Setting up parameters.")

parameters = {}

# ODMD Parameters
num_timesteps = 100

# COMPUTATION PARAMETERS
computation_type = 'S'  # computation_type OPTIONS: Classical, Simulation, Hardware, Job

# Generic System Parameters
parameters['sites']    = 3
parameters['Dt']       = .1
parameters['shots']    = 10**5
parameters['scaling']  = 3/4*pi
parameters['shifting'] = 0
parameters['overlap']  = 0.75   # the initial state overlap

# SPECIFIC SYSTEM TYPE
parameters['system']     = 'TFI' # OPTIONS: TFIM, SPIN, HUBBARD, H_2

# Transverse Field Ising Model Parameters
parameters['g'] = 4 # magnetic field strength (TFIM)
parameters['method_for_model'] = 'F' # OPTIONS: F3C, Qiskit
parameters['trotter'] = 10 # only with method_for_model = F3C

# Spin Model Parameters
parameters['J'] = 4 # coupling strength (SPIN)

# Hubbard Parameters
parameters['t'] = 1 # left-right hopping (HUBB)
parameters['U'] = 1 # up-down hopping (HUBB)
parameters['x'] = 3 # x size of latice (HUBB)
parameters['y'] = 1 # y size of latice (HUBB)

# H_2 Parameters
parameters['distance'] = .5

# PREPROCESSING

computation_type     = computation_type[0].upper()
parameters['system'] = parameters['system'][0:3].upper()
time_step = parameters['Dt']

# parameter checking (if there's an error change parameters in question)
assert(computation_type == 'C' or computation_type == 'S' or computation_type == 'H' or computation_type == 'J')
assert(parameters['system'] == "TFI" or parameters['system'] == "SPI" or parameters['system'] == "HUB" or parameters['system'] == "H_2")
if 'overlap' not in parameters: parameters['overlap'] = 1
assert(0<=parameters['overlap']<=1)

# verify system parameters are setup correctly
if parameters['system'] == "TFI":
    parameters['method_for_model'] = parameters['method_for_model'][0].upper()
    assert(parameters['method_for_model']=="F" or parameters['method_for_model']=="Q")
elif parameters['system'] == 'HUB':
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
elif parameters['system'] == 'SPI':
    assert(parameters['J']!=0)
elif parameters['system'] == 'H_2':
    parameters['sites']=1

# backend setup
if computation_type == 'H' or computation_type == 'J':
    backend = create_hardware_backend()
else:
    backend = AerSimulator(noise_model = NoiseModel())
sampler = Sampler(backend)

# define a system for naming files
def make_filename(computation_type, num_timesteps, parameters, add_shots = False):
    system = parameters['system']
    string = "comp="+computation_type+"_sys="+system
    string+="_n="+str(parameters['sites'])
    if system=="TFI":
        if computation_type != 'C':
            method_for_model = parameters['method_for_model']
            string+="_m="+method_for_model
            if method_for_model == 'F':
                string+="_trotter="+str(parameters['trotter'])
        string+="_g="+str(parameters['g'])
    elif system=="SPI":
        string+="_J="+str(parameters['J'])
    elif system=="HUB":
        string+="_t="+str(parameters['t'])
        string+="_U="+str(parameters['U'])
        string+="_x="+str(parameters['x'])
        string+="_y="+str(parameters['y'])
    elif system=="H_2":
        string+="_dist="+str(parameters['distance'])
    string+="_scale="+str(parameters['scaling'])
    string+="_shift="+str(parameters['shifting'])
    string+="_overlap="+str(parameters['overlap'])
    string+="_Dt="+str(parameters['Dt'])
    string += "_maxitr="+str(num_timesteps)
    if add_shots: string += "_shots="+str(parameters['shots'])
    return string

print("Parameters are setup.")