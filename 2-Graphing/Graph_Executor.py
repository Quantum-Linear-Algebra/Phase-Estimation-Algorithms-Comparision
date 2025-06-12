import sys 
paths = ['.', './0-Data', './1-Algorithms', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from numpy import pi

import Parameters as param
from Parameters import make_filename
import Data_Generator as data
import Algorithm_Executor as algo
import Graph_Generator as graph_gen

parameters = {}
# NOTE: Specifying unused parameters will not affect computation with the used parameters

# Generic Parameters
parameters['comp_type']     = 'S' # OPTIONS: Classical, Simulation, Hardware, Job
parameters['num_timesteps'] = 1000
parameters['sites']         = 3
parameters['Dt']            = 0.05
parameters['shots']         = 10**2
parameters['scaling']       = 3/4*pi
parameters['shifting']      = 0
parameters['overlap']       = 0.75   # the initial state overlap

# SPECIFIC SYSTEM TYPE
parameters['system']     = 'HUB' # OPTIONS: TFIM, SPIN, HUBBARD, H_2

# Transverse Field Ising Model Parameters
parameters['g'] = 4 # magnetic field strength (TFIM)
parameters['method_for_model'] = 'Q' # OPTIONS: F3C, Qiskit
parameters['trotter'] = 1000 # only with method_for_model = F3C

# Spin Model Parameters
parameters['J'] = 4 # coupling strength (SPIN)

# Hubbard Parameters
parameters['t'] = 1 # left-right hopping (HUBB)
parameters['U'] = 1 # up-down hopping (HUBB)
parameters['x'] = 3 # x size of latice (HUBB)
parameters['y'] = 1 # y size of latice (HUBB)

# H_2 Parameters
parameters['distance'] = .5

# Algorithms to use
parameters['algorithms'] = ['ODMD'] # OPTIONS: ODMD, VQPE, QCELS, ML-QCELS

backend = param.check(parameters)

graph_gen.run(parameters)