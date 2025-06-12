import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from scipy.linalg import eigh
from scipy.fft import fft, fftshift, fftfreq

paths = './..', './../0-Data'
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from Parameters import make_filename
from Data_Generator_Helper import create_hamiltonian, make_overlap

def run(parameters):
    filename = make_filename(parameters, add_shots=True)+".pkl"
    with open("0-Data/Expectation_Values/"+filename, 'rb') as file:
        exp_vals = pickle.load(file)
        
    with open("1-Algorithms/Results/"+filename, 'rb') as file:
        [observables, est_E_0s] = pickle.load(file)

    H,real_E_0 = create_hamiltonian(parameters)
    E,vecs = eigh(H)
    vecs = [vecs[:,i] for i in range(len(vecs))]
    sv = make_overlap(vecs[0], parameters['overlap'])

    plt.title("Overlap")
    plt.bar(range(len(vecs)),[np.linalg.norm(sv@vecs[i]) for i in range(len(vecs))], width=.6)
    plt.xticks(range(len(vecs)), [f'{i:.3}' for i in E])
    plt.show()
    print(E)

    plt.figure()
    plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in exp_vals], label = "Re")
    plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in exp_vals], label = "Im")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Real Expectation Value")
    plt.title("Expectation Value with Dt="+str(parameters['Dt'])+" and overlap="+str(parameters['overlap']))
    plt.savefig("2-Graphing/Graphs/"+make_filename(parameters, add_shots=True)+"_Expectation_Value.png")
    plt.show()


    plt.figure()
    plt.plot(fftshift(fftfreq(len(exp_vals), d=parameters['Dt'])), abs(fftshift(fft(exp_vals))), label = 'FFT')
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Fourier Transform of Expectation Value with Dt="+str(parameters['Dt'])+" and overlap="+str(parameters['overlap']))
    plt.savefig("2-Graphing/Graphs/"+make_filename(parameters)+"_Fourier_Transform_Expectation_Value.png")
    plt.show()

    plt.figure()
    use_shots = False
    if use_shots: total_shots = [w*parameters['shots'] for w in observables]
    err = [abs(w-real_E_0) for w in est_E_0s]
    label = 'Dt = '+str(parameters['Dt'])
    if use_shots: x=total_shots 
    else: x=observables
    plt.plot(x, err, label = label)
    plt.title("Calculated Absolute Error in Energy for "+parameters['system']+" as ODMD Iterates")
    plt.ylabel("Absolute Error")
    if use_shots: plt.xlabel("Total Shots")
    else: plt.xlabel("Number of Observables")
    plt.legend()
    # plt.xlim([0,10])
    # plt.ylim([0,10])
    plt.yscale('log')
    plt.savefig("2-Graphing/Graphs/"+make_filename(parameters)+"_Abs_Error.png")
    plt.show()

    plt.figure()
    label = 'ODMD Dt = '+str(parameters['Dt'])
    eigs = np.linalg.eigvals(H)
    eigs = [(eig.real-parameters['shifting'])*parameters['r_scaling'] for eig in eigs]
    for i in range(len(eigs)):
        plt.plot([x[0],x[-1],], [eigs[i],eigs[i]], label = "E"+str(i))
    plt.plot(x, est_E_0s, label = label)
    plt.legend()
    # plt.ylim(eigs[0]+.1, eigs[0]-.1)
    plt.title("Convergence in Energy for "+parameters['system']+" as ODMD Iterates")
    plt.ylabel("Eigenvalue")
    plt.savefig("2-Graphing/Graphs/"+make_filename(parameters)+"_Convergence.png")
    plt.show()