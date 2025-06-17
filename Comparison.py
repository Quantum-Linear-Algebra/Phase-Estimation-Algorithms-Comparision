import sys 
paths = ['./0-Data', './1-Algorithms', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from numpy import pi

import Parameters as param
import Data_Manager as data
import Algorithm_Manager as algo
import Graph_Manager as graph_gen

parameters = {}
# NOTE: Specifying unused parameters will not affect computation with the used parameters

# Generic Parameters
parameters['comp_type']     = 'S' # OPTIONS: Classical, Simulation, Hardware, Job
parameters['num_timesteps'] = 100
parameters['sites']         = 3
parameters['Dt']            = 0.01
parameters['shots']         = 100
parameters['scaling']       = 3/4*pi
parameters['shifting']      = 0
parameters['overlap']       = 1   # the initial state overlap

# SPECIFIC SYSTEM TYPE
parameters['system']     = 'TFI' # OPTIONS: TFIM, SPIN, HUBBARD, H_2

# Transverse Field Ising Model Parameters
parameters['g'] = 4 # magnetic field strength (TFIM)
parameters['method_for_model'] = 'Q' # OPTIONS: F3C, Qiskit
parameters['trotter'] = 10 # only with method_for_model = F3C

# Spin Model Parameters
parameters['J'] = 4 # coupling strength (SPIN)

# Hubbard Parameters
parameters['t'] = 1 # left-right hopping (HUBB)
parameters['U'] = 10 # up-down hopping (HUBB)
parameters['x'] = 3 # x size of latice (HUBB)
parameters['y'] = 1 # y size of latice (HUBB)

# H_2 Parameters
parameters['distance'] = .5

# Algorithms
parameters['algorithms'] = ['VQPE'] # OPTIONS: ODMD, VQPE, UVQPE, QCELS, ML-QCELS
parameters['fourier_filtering'] = False

backend = param.check(parameters)

if __name__ == "__main__":
    parameters = data.run(parameters, backend)
    algo.run(parameters, skipping=1)
    graph_gen.run(parameters)