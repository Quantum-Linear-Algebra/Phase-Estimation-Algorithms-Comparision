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
from Data_Manager import create_hamiltonian

def run(parameters, max_itr=-1):
    reruns = parameters['reruns']
    # get related data
    try: os.mkdir('2-Graphing/Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)+'.pkl'
        if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
            with open('0-Data/Expectation_Values/linear_'+filename, 'rb') as file:
                all_exp_vals = pickle.load(file)
    except: print("Failed to grab expectation value data. Try generating the dataset."); sys.exit(0)
    all_est_E_0s = []
    all_observables = []
    for algo in parameters['algorithms']:    
        try:
            with open('1-Algorithms/Results/'+algo+'_'+filename, 'rb') as file:
                [algo_observables, algo_est_E_0s] = pickle.load(file)
            all_est_E_0s.append(algo_est_E_0s)
            all_observables.append(algo_observables)
        except: print('Failed to grab energy estimates for'+algo+'. Try recalculating the results of the algorithm.'); sys.exit(0)
    
    H,real_E_0 = create_hamiltonian(parameters)
    E,vecs = eigh(H)
    # real_E_0 = E[0]
    vecs = [vecs[:,i] for i in range(len(vecs))]
    sv = parameters['sv']

    # create related graphs
    plt.figure()
    plt.title('Overlap')
    plt.bar(range(len(vecs)),[np.abs(sv@vecs[i])**2 for i in range(len(vecs))], width=.6)
    plt.xticks(range(len(vecs)), [f'{i:.5}' for i in E], rotation=90)
    plt.xlabel('Energy value of eigenstate', labelpad=10)
    plt.ylabel('Overlap with input state')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Spectrum.png', bbox_inches='tight')
    plt.show()

    alpha = 1/len(all_est_E_0s[0])
    if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
        plt.figure()
        avg_exp_vals = np.zeros(len(all_exp_vals[0]), dtype=complex)
        for i in range(len(all_exp_vals)):
            exp_vals = all_exp_vals[i]
            avg_exp_vals += exp_vals
            plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in exp_vals], alpha = alpha, c = 'orange')
            plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in exp_vals], alpha = alpha, c = 'blue')
        avg_exp_vals /= len(all_exp_vals)
        plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in avg_exp_vals], c = 'orange', label = 'Real')
        plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in avg_exp_vals], c = 'blue', label = 'Imaginary')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.title('Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots=True)+'_Expectation_Value.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        for i in range(len(all_exp_vals)):
            exp_vals = all_exp_vals[i]
            plt.plot(fftshift(fftfreq(len(exp_vals), d=parameters['Dt'])), abs(fftshift(fft(exp_vals))), c = 'purple', alpha = alpha)
        plt.plot(fftshift(fftfreq(len(avg_exp_vals), d=parameters['Dt'])), abs(fftshift(fft(avg_exp_vals))), c = 'purple', label = 'FFT')
        
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Fourier_Transform_Expectation_Value.png', bbox_inches='tight')
        plt.show()
    colors = {'QCELS':'red', 'ODMD':'blue', 'ML_QCELS':'orange', 'UVQPE':'limegreen', 'VQPE':'darkolivegreen'}

    use_shots = False
    xs = []
    for i in range(len(all_est_E_0s)):
        observables = all_observables[i][0]
        if use_shots: total_shots = [w*parameters['shots'] for w in observables]
        if use_shots: x=total_shots 
        else: x=observables
        xs.append(x)
    
    longest_x = 0
    for x in xs:
        num = x[-1]
        if longest_x < num: longest_x = num

    averages = {}
    for i in range(len(parameters['algorithms'])):
        algo = parameters['algorithms'][i]
        color = colors[algo]
        avg_est_E_0s = np.zeros(len(all_est_E_0s[i][0]), dtype=complex)
        for j in range(len(all_est_E_0s[i])):
            est_E_0s = all_est_E_0s[i][j]
            avg_est_E_0s += est_E_0s
        avg_est_E_0s /= len(all_est_E_0s[i])
        averages[algo] = avg_est_E_0s

    
    plt.figure()
    for i in range(len(all_est_E_0s)):
        algo = parameters['algorithms'][i]
        color = colors[algo]
        x = xs[i]
        for j in range(len(all_est_E_0s[i])):
            est_E_0s = all_est_E_0s[i][j]
            err = [abs(w-real_E_0) for w in est_E_0s]
            plt.plot(x, err, c = color, alpha = alpha)
        avg_est_E_0s = averages[algo]
        err = [abs(w-real_E_0) for w in avg_est_E_0s]
        plt.plot(x, err, c = color, label = algo)
        
        
    plt.plot([0,longest_x], [10**-3, 10**-3], label = 'Chemical Accuracy', c = 'black')
    if max_itr != -1: plt.xlim([0, max_itr])
    plt.title('Convergence Absolute Error in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
    plt.ylabel('Absolute Error')
    if use_shots: plt.xlabel('Total Shots')
    else: plt.xlabel('Number of Observables')
    plt.legend()
    # plt.xlim([0,10])
    # plt.ylim([0,10])
    plt.yscale('log')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Abs_Error.png', bbox_inches='tight')
    plt.show()

    plt.figure()
    for i in range(len(all_est_E_0s)):
        algo = parameters['algorithms'][i]
        color = colors[algo]
        for j in range(len(all_est_E_0s[i])):
            plt.plot(xs[i], all_est_E_0s[i][j], c = color, alpha = alpha)
        plt.plot(xs[i], averages[algo], c = color, label = algo)
    eigs = np.linalg.eigvals(H)
    eigs = np.sort([(eig.real-parameters['shifting'])*parameters['r_scaling'] for eig in eigs])
    for i in range(len(eigs)):
        plt.plot([0,longest_x], [eigs[i],eigs[i]], ':', label = 'E'+str(i))
    if max_itr != -1: plt.xlim([0, max_itr])
    if use_shots: plt.xlabel('Total Shots')
    else: plt.xlabel('Number of Observables')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    dis = (eigs[2]-eigs[0])/2
    # plt.ylim(eigs[0]-dis, eigs[2]+dis)
    plt.title('Convergence in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
    plt.ylabel('Eigenvalue')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Convergence.png', bbox_inches='tight')
    plt.show()
    isolate_graphs(parameters)

def isolate_graphs(parameters):
    print('Attempting to copy newly generated graphs.')
    try: os.mkdir('Recent_Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)
        graph_types = ['Spectrum', 'Abs_Error', 'Convergence']
        if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
            graph_types.append('Expectation_Value')
            graph_types.append('Fourier_Transform_Expectation_Value')
        for graph_type in graph_types:
            exit_code = os.system('cp 2-Graphing/Graphs/'+filename+'_'+graph_type+'.png Recent_Graphs/'+graph_type+'.png')
            assert(exit_code==0)
        print('Successfully copied newly generated graphs. (', end ='')
        for graph_type in graph_types[:len(graph_types)-1]:
            print('Recent_Graphs/'+graph_type+'.png, ', end='') 
        print('Recent_Graphs/'+graph_types[-1]+'.png)') 
    except Exception as e:
        print(e)
        print('One or more of the regularly generated graphs does not exist. Try regenerating the desired graphs.')

if __name__ == '__main__':
    import sys 
    paths = ['.', './2-Graphing']
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    from Comparison import parameters
    from Parameters import check
    check(parameters)
    run(parameters)
    # isolate_graphs(parameters)