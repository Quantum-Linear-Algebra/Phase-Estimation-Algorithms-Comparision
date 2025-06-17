import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, os
from scipy.linalg import eigh
from scipy.fft import fft, fftshift, fftfreq

paths = '.', './0-Data'
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from Parameters import make_filename
from Data_Manager import create_hamiltonian, make_overlap

def run(parameters, max_itr=-1):
    try: os.mkdir('2-Graphing/Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)+'.pkl'
        with open('0-Data/Expectation_Values/'+filename, 'rb') as file:
            exp_vals = pickle.load(file)
    except: print("Failed to grab expectation value data. Try generating the dataset")
    all_est_E_0s = []
    all_observables = []
    for algo in parameters['algorithms']:    
        try:
            with open('1-Algorithms/Results/'+algo+'_'+filename, 'rb') as file:
                [observables, est_E_0s] = pickle.load(file)
            all_est_E_0s.append(est_E_0s)
            all_observables.append(observables)
        except: print('Failed to grab energy estimates for'+algo+'. Try recalculating the results the algorithm.')
    H,real_E_0 = create_hamiltonian(parameters)
    E,vecs = eigh(H)
    vecs = [vecs[:,i] for i in range(len(vecs))]
    sv = make_overlap(vecs[0], parameters['overlap'])

    plt.title('Overlap')
    plt.bar(range(len(vecs)),[np.linalg.norm(sv@vecs[i]) for i in range(len(vecs))], width=.6)
    plt.xticks(range(len(vecs)), [f'{i:.3}' for i in E])
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Spectrum.png')
    plt.show()

    plt.figure()
    plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in exp_vals], label = 'Re')
    plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in exp_vals], label = 'Im')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Real Expectation Value')
    plt.title('Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots=True)+'_Expectation_Value.png')
    plt.show()


    plt.figure()
    plt.plot(fftshift(fftfreq(len(exp_vals), d=parameters['Dt'])), abs(fftshift(fft(exp_vals))), label = 'FFT')
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Fourier_Transform_Expectation_Value.png')
    plt.show()
    
    plt.figure()
    use_shots = False
    xs = []
    for i in range(len(all_est_E_0s)):
        algo = parameters['algorithms'][i]
        observables = all_observables[i]
        est_E_0s = all_est_E_0s[i]
        if use_shots: total_shots = [w*parameters['shots'] for w in observables]
        err = [abs(w-real_E_0) for w in est_E_0s]
        if use_shots: x=total_shots 
        else: x=observables
        xs.append(x)
        plt.plot(x, err, label = algo)
    if max_itr != -1: plt.xlim([0, max_itr])
    plt.title('Convergence Absolute Error in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
    plt.ylabel('Absolute Error')
    if use_shots: plt.xlabel('Total Shots')
    else: plt.xlabel('Number of Observables')
    plt.legend()
    # plt.xlim([0,10])
    # plt.ylim([0,10])
    plt.yscale('log')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Abs_Error.png')
    plt.show()

    plt.figure()
    eigs = np.linalg.eigvals(H)
    eigs = [(eig.real-parameters['shifting'])*parameters['r_scaling'] for eig in eigs]
    for i in range(len(eigs)):
        plt.plot([x[0],x[-1],], [eigs[i],eigs[i]], label = 'E'+str(i))
    for i in range(len(all_est_E_0s)):
        plt.plot(xs[i], all_est_E_0s[i], label = parameters['algorithms'][i])
    if max_itr != -1: plt.xlim([0, max_itr])
    plt.legend()
    # plt.ylim(eigs[0]+.1, eigs[0]-.1)
    plt.title('Convergence in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
    plt.ylabel('Eigenvalue')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Convergence.png')
    plt.show()
    isolate_graphs(parameters)

def isolate_graphs(parameters):
    print('Attempting to copy newly generated graphs.')
    try: os.mkdir('Recent_Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)
        graph_types = ['Spectrum', 'Expectation_Value', 'Fourier_Transform_Expectation_Value', 'Abs_Error', 'Convergence']
        for graph_type in graph_types:
            exit_code = os.system('cp 2-Graphing/Graphs/'+filename+'_'+graph_type+'.png Recent_Graphs/'+graph_type+'.png')
            assert(exit_code==0)
        print('Successfully copied newly generated graphs. (Recent_Graphs/*.png)')
    except:
        print('One or more of the regularly generated graphs do not exist. Try regenerating the desired graphs.')

if __name__ == '__main__':
    import sys 
    paths = ['.', './2-Graphing']
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    from Comparison import parameters
    from Parameters import check
    check(parameters)
    run(parameters, max_itr=50)
    # isolate_graphs(parameters)