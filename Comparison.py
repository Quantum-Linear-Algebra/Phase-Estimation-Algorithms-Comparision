

from numpy import pi

import Parameters as param

import sys 
paths = ['./0-Data', './1-Algorithms', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
import Data_Manager as data
import Algorithm_Manager as algo
import Graph_Manager as graph_gen

parameters = {}
# NOTE: Specifying unused parameters will not affect computation with the used parameters

# Generic Parameters
parameters['comp_type']    = 'C' # OPTIONS: Classical, Simulation, Hardware, Job
parameters['observables']  = 110
parameters['sites']        = 2
parameters['T']            = 10000000
parameters['shots']        = 10**2
parameters['scaling']      = 3/4*pi
parameters['shifting']     = 0
parameters['overlap']      = 1   # the initial state overlap
# parameters['distribution'] = [.4,.4]+[(1-.8)/(2**8-2)]*(2**8-2)

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
parameters['x'] = 2 # x size of latice (HUBB)
parameters['y'] = 1 # y size of latice (HUBB)

# H_2 Parameters
parameters['distance'] = .5

# Algorithm Paramters
parameters['algorithms']    = ['ML_QCELS'] # ALGORITHMS: 'ODMD', 'FODMD', 'VQPE', 'UVQPE', 'QCELS', 'ML_QCELS', 'QMEGS'
parameters['const_obs']     = False # if False then constant time
parameters['num_time_sims'] = 1
parameters['reruns']        = 1

# Algorithm Specific Parameters
parameters['ODMD_svd_threshold']   = 10**-1
parameters['ODMD_full_observable'] = True

parameters['FODMD_svd_threshold']   = 10**-1
parameters['FODMD_full_observable'] = True
parameters['FODMD_gamma_range']     = (1,3) # (min, max)
parameters['FODMD_filter_count']    = 6

parameters['VQPE_svd_threshold']   = 10**-1

parameters['UVQPE_svd_threshold']  = 9*10**-1

parameters['ML_QCELS_time_steps']  = 5
parameters['ML_QCELS_calc_Dt']     = False

parameters['QMEGS_sigma']          = 1
parameters['QMEGS_q']              = 0.05
parameters['QMEGS_alpha']          = 5
parameters['QMEGS_K']              = 2

if __name__ == "__main__":
    returns = param.check(parameters)
    data.run(parameters, returns)
    algo.run(parameters, skipping=10)
    graph_gen.run(parameters, show_std=True)