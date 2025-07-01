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

def run(parameters, max_itr=-1, skipping=1, show_std=False, use_shots=False):
    # setup relavant variables
    reruns = parameters['reruns']
    fourier_filtering = parameters['fourier_filtering']
    only_ML_QCLES = parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS'] 
    if fourier_filtering:
        filter_count = parameters['filter_count']
        gammas = np.linspace(parameters['gamma_range'][0], parameters['gamma_range'][1], parameters['filter_count'])
    # get related data
    filename = make_filename(parameters, add_shots=True)+'.pkl'
    try: os.mkdir('2-Graphing/Graphs')
    except: pass
    if not only_ML_QCLES:
        try:
            with open('0-Data/Expectation_Values/linear_'+filename, 'rb') as file:
                all_exp_vals = pickle.load(file) # reruns, exp_vals
        except: print("Failed to grab expectation value data. Try generating the dataset."); sys.exit(0)
        if parameters['fourier_filtering']:
            filename = make_filename(parameters, fourier_filtered=True, add_shots=True)+'.pkl'
            with open('0-Data/Expectation_Values/Denoised/linear_'+filename, 'rb') as file:
                all_ff_exp_vals = pickle.load(file) # reruns, filters, exp_vals

    all_est_E_0s = []
    all_observables = []
    for algo in parameters['algorithms']:    
        try:
            with open('1-Algorithms/Results/'+algo+'_'+filename, 'rb') as file:
                [algo_observables, algo_est_E_0s] = pickle.load(file)
            all_est_E_0s.append(algo_est_E_0s)
            all_observables.append(algo_observables)
        except: print('Failed to grab energy estimates for '+algo+'. Try recalculating the results of the algorithm.'); sys.exit(0)
    
    H,real_E_0 = create_hamiltonian(parameters)
    E,vecs = eigh(H)
    # real_E_0 = E[0]
    vecs = [vecs[:,i] for i in range(len(vecs))]
    sv = parameters['sv']

    # check lengths of data
    if not only_ML_QCLES:
        if reruns > len(all_exp_vals):
            print('Number of linear time series is too small. Reducing reruns.')
            reruns=len(all_exp_vals)
    if fourier_filtering:
        if reruns > len(all_ff_exp_vals):
            print('Number of fourier filtered time series is too small. Reducing reruns.')
            reruns=len(all_ff_exp_vals)
        for ff_exp_vals in all_ff_exp_vals:
            if filter_count > len(ff_exp_vals):
                print('Less filters than expected. Reducing filter count.')
                filter_count = len(ff_exp_vals)
    for i in range(len(parameters['algorithms'])):
        if reruns > len(all_est_E_0s[i]):
            print('Number of ground state estimations is too small for '+parameters['algorithms'][i]+'. Reducing reruns.')
            reruns=len(all_est_E_0s[i])
        if reruns > len(all_observables[i]):
            print('Number of observables is too small for '+parameters['algorithms'][i]+'. Reducing reruns.')
            reruns=len(all_observables[i]) 

    # create related graphs
    plt.figure()
    plt.title('Overlap')
    plt.bar(range(len(vecs)),[np.abs(sv@vecs[i])**2 for i in range(len(vecs))], width=.6)
    plt.xticks(range(len(vecs)), [f'{i:.5}' for i in E], rotation=90)
    plt.xlabel('Energy value of eigenstate', labelpad=10)
    plt.ylabel('Overlap with input state')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Spectrum.png', bbox_inches='tight')
    plt.show()

    alpha = 1/reruns
    if not only_ML_QCLES:
        plt.figure()
        avg_exp_vals = np.zeros(len(all_exp_vals[0]), dtype=complex)
        for i in range(reruns):
            exp_vals = all_exp_vals[i]
            avg_exp_vals += exp_vals
            plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in exp_vals], alpha = alpha, c = 'orange')
            plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in exp_vals], alpha = alpha, c = 'blue')
        avg_exp_vals /= reruns
        plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in avg_exp_vals], c = 'orange', label = 'Real')
        plt.plot([i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in avg_exp_vals], c = 'blue', label = 'Imaginary')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.title('Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots=True)+'_Expectation_Value.png', bbox_inches='tight')
        plt.show()
        

        plt.figure()
        for i in range(reruns):
            exp_vals = all_exp_vals[i]
            plt.plot(fftshift(fftfreq(len(exp_vals), d=parameters['Dt'])), abs(fftshift(fft(exp_vals))), c = 'purple', alpha = alpha)
        plt.plot(fftshift(fftfreq(len(avg_exp_vals), d=parameters['Dt'])), abs(fftshift(fft(avg_exp_vals))), c = 'purple', label = 'FFT')
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Fourier_Transform_Expectation_Value.png', bbox_inches='tight')
        plt.show()

        if fourier_filtering:
            plt.figure()
            ax = plt.axes(projection='3d') 
            for i in range(reruns):
                for j in range(filter_count):
                    exp_vals = all_ff_exp_vals[i][j]
                    plt.plot(gammas[j], [i*parameters['Dt'] for i in range(len(exp_vals))], [i.real for i in exp_vals], alpha = alpha, c = 'orange')
                    plt.plot(gammas[j], [i*parameters['Dt'] for i in range(len(exp_vals))], [i.imag for i in exp_vals], alpha = alpha, c = 'blue')
            plt.ylabel('Time')
            ax.set_zlabel('Expectation Value')
            plt.xlabel('gamma')
            plt.title('Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
            plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, fourier_filtered=True, add_shots=True)+'_Expectation_Value.png')
            plt.show()
            
            plt.figure()
            ax = plt.axes(projection='3d') 
            for i in range(reruns):
                for j in range(filter_count):
                    exp_vals = all_ff_exp_vals[i][j]
                    plt.plot(gammas[j], fftshift(fftfreq(len(exp_vals), d=parameters['Dt'])), abs(fftshift(fft(exp_vals))))
            plt.ylabel('Frequency')
            ax.set_zlabel('Amplitude')
            plt.xlabel('gamma')
            plt.title('Fourier Transform of Expectation Value with Dt='+str(parameters['Dt'])+' and overlap='+str(parameters['overlap']))
            plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, fourier_filtered=True, add_shots =True)+'_Fourier_Transform_Expectation_Value.png')
            plt.show()
    
    if parameters['algorithms']: # if theres at least one algorithm
        colors = {'QCELS':'red', 'ODMD':'blue', 'ML_QCELS':'orange', 'UVQPE':'limegreen', 'VQPE':'darkolivegreen'}
        shapes = {'QCELS':'o', 'ODMD':'^', 'ML_QCELS':'X', 'UVQPE':'P', 'VQPE':'*'}

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
        
        plt.figure()
        avg_err = []
        for i in range(len(all_est_E_0s)):
            algo = parameters['algorithms'][i]
            color = colors[algo]
            shape = shapes[algo]
            x = xs[i]
            avg_err.append(np.zeros(len(all_est_E_0s[i][0])))
            for j in range(len(all_est_E_0s[i])):
                est_E_0s = all_est_E_0s[i][j]
                err = [abs(w-real_E_0) for w in est_E_0s]
                avg_err[i] += err
                # plt.scatter(x, err, c = color, alpha = alpha)
            avg_err[i] /= len(all_est_E_0s[i])
            # for j in range(len(avg_err[i])):
            plt.plot(x, avg_err[i], c = color, marker = shape, label = algo)

        if show_std:
            for i in range(len(all_est_E_0s)):
                algo = parameters['algorithms'][i]
                color = colors[algo]
                std_err = []
                for j in range(len(all_est_E_0s[i][0])):
                    tmp = []
                    for k in range(reruns):
                        tmp.append(abs(all_est_E_0s[i][k][j] - real_E_0))
                    std_err.append(np.std(tmp))
                plt.fill_between(xs[i], avg_err[i] - std_err,avg_err[i] + std_err, color=color, alpha=0.2)

        plt.plot([0,longest_x], [10**-3, 10**-3], label = 'Chemical Accuracy', c = 'black')
        if max_itr != -1: plt.xlim([0, max_itr])
        plt.title('Convergence Absolute Error in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
        plt.ylabel('Absolute Error')
        if use_shots: plt.xlabel('Total Shots')
        else: plt.xlabel('Number of Observables')
        plt.legend()
        # plt.xlim([0,10])
        # plt.ylim([0,0.00001])
        plt.yscale('log')
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Abs_Error.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        avg_E_0s = []
        for i in range(len(all_est_E_0s)):
            algo = parameters['algorithms'][i]
            color = colors[algo]
            shape = shapes[algo]
            avg_E_0s.append(np.zeros(len(all_est_E_0s[i][0])))
            for j in range(len(all_est_E_0s[i])):
                # plt.scatter(xs[i], all_est_E_0s[i][j], c = color, alpha = alpha)
                avg_E_0s[i] += all_est_E_0s[i][j]
            avg_E_0s[i] /= len(all_est_E_0s[i])
            plt.plot(xs[i], avg_E_0s[i], c = color, marker = shape, label = algo)
        
        if show_std:
            for i in range(len(all_est_E_0s)):
                algo = parameters['algorithms'][i]
                color = colors[algo]
                std_exp_vals = []
                for j in range(len(all_est_E_0s[i][0])):
                    tmp = []
                    for k in range(reruns):
                        tmp.append(all_est_E_0s[i][k][j])
                    std_exp_vals.append(np.std(tmp))
                plt.fill_between(xs[i], avg_E_0s[i] - std_exp_vals, avg_E_0s[i] + std_exp_vals, color=color, alpha=0.2)

        eigs = np.linalg.eigvals(H)
        eigs = np.sort([(eig.real-parameters['shifting'])*parameters['r_scaling'] for eig in eigs])
        for i in range(len(eigs)):
            plt.plot([0,longest_x], [eigs[i],eigs[i]], ':', label = 'E'+str(i))
        if max_itr != -1: plt.xlim([0, max_itr])
        if use_shots: plt.xlabel('Total Shots')
        else: plt.xlabel('Number of Observables')
        plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        # dis = (eigs[2]-eigs[0])/2
        # plt.ylim(eigs[0]-dis, eigs[2]+dis)
        # plt.ylim(eigs[0] - 0.1, eigs[0] + 0.1)
        plt.title('Convergence in Energy for '+parameters['system']+' with overlap='+str(parameters['overlap']))
        plt.ylabel('Eigenvalue')
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Convergence.png', bbox_inches='tight')
        plt.show()
    
    isolate_graphs(parameters)

def isolate_graphs(parameters):
    exit_code = os.system('rm -rf Recent_Graphs')
    assert(exit_code == 0)
    print('Attempting to copy newly generated graphs.')
    try: os.mkdir('Recent_Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)
        graph_types = ['Spectrum']
        if parameters['algorithms']:
            graph_types.append('Abs_Error')
            graph_types.append('Convergence')
        if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
            graph_types.append('Expectation_Value')
            graph_types.append('Fourier_Transform_Expectation_Value')
        for graph_type in graph_types:
            exit_code = os.system('cp 2-Graphing/Graphs/'+filename+'_'+graph_type+'.png Recent_Graphs/'+graph_type+'.png')
            assert(exit_code==0)
            if parameters['fourier_filtering'] and graph_type[-17:] == 'Expectation_Value':
                fn = make_filename(parameters, fourier_filtered=True, add_shots=True)
                exit_code = os.system('cp 2-Graphing/Graphs/'+fn+'_'+graph_type+'.png Recent_Graphs/Fourier_Filtered_'+graph_type+'.png')
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
    run(parameters, show_std=True)
    # isolate_graphs(parameters)