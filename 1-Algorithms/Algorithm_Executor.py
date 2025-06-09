from ODMD import ODMD
import pickle
import sys
sys.path.append('.')
from Parameters import *
sys.path.append('./0-Data')
import Data_Generator

def run(parameters):
    num_timesteps = parameters['num_timesteps']
    filename = make_filename(parameters, add_shots=True)+".pkl"
    with open("0-Data/Expectation_Values/"+filename, 'rb') as file:
        exp_vals = pickle.load(file)

    # Algorithmic Parameters
    svd_threshold = 10**-3

    results = []
    print()
    Dt = parameters['Dt']
    print("Using data from time step:", Dt)
    est_E_0s = []
    est_E_0s = ODMD(exp_vals, Dt, svd_threshold, num_timesteps, show_steps=False) 
    # readjust energy to what it originially was
    for i in range(len(est_E_0s)):
        est_E_0s[i] = (est_E_0s[i]-parameters['shifting'])*parameters['r_scaling']

    with open("1-Algorithms/Results/"+filename, 'wb') as file:
        pickle.dump([range(num_timesteps), est_E_0s], file)