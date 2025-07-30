import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, os
from scipy.linalg import eigh
from scipy.fft import fft, fftshift, fftfreq

paths = '.', './0-Data', './1-Algorithms/Algorithms'
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from Parameters import make_filename, check_contains_linear
from ODMD import fourier_filter_exp_vals

def run(parameters, max_itr=-1, skipping=1, show_std=False):
    print('\nGenerating Graphs')
    # setup relavant variables
    reruns = parameters['reruns']
    
    contains_linear = check_contains_linear(parameters['algorithms'])
    fourier_filtering = 'FDODMD' in parameters['algorithms']
    # Dt = parameters['T']/parameters['observables']

    # if 'overlap' in parameters:
    #     spectrum_string = 'overlap='+str(parameters['overlap'])
    # elif 'distribution' in parameters:
    #     spectrum_string = 'distribution='
    #     for i in parameters['distribution'][:3][:-1]:
    #         spectrum_string+=f'{i:0.2},'
    #     var = parameters['distribution'][3]
    #     spectrum_string +=f'{var:0.2}...'
    
    if fourier_filtering:
        gamma_range = parameters['algorithms']['FDODMD']['gamma_range']
        filters = parameters['algorithms']['FDODMD']['filter_count']
        gammas = np.linspace(gamma_range[0], gamma_range[1], filters)
    
    # get related data
    try: os.mkdir('2-Graphing/Graphs')
    except: pass
    if contains_linear:
        linear_ts = None
        for ts in parameters['time_series']:
            linear_ts = ts
            if ts[0] == 'linear':
                (time_series_name, T, observables, shots, full_observable) = ts
                filename = make_filename(parameters, key=time_series_name, add_shots=True, T=T, obs=observables, shots=shots, fo=full_observable)+'.pkl'
                with open('0-Data/Expectation_Values/'+filename, 'rb') as file:
                    all_exp_vals = pickle.load(file) # reruns, exp_vals
        try:
            with open('0-Data/Expectation_Values/'+filename, 'rb') as file:
                all_exp_vals = pickle.load(file) # reruns, exp_vals
        except Exception as e:
            print(e)
            print("Failed to grab expectation value data. Try regenerating the dataset.")


    

    all_queries = {}
    all_est_E_0s = {}
    for ts in parameters['time_series']:
        (time_series_name, T, observables, shots, full_observable) = ts
        algos = parameters['time_series'][ts]
        for algo in algos:    
            all_queries[algo] = {}
            all_est_E_0s[algo] = {}
            try:
                with open('1-Algorithms/Results/'+algo+'_'+make_filename(parameters, add_shots=True, T=T, obs=observables, fo=full_observable, shots=shots)+'.pkl', 'rb') as file:
                    [algo_queries, algo_est_E_0s] = pickle.load(file)
                all_queries[algo][T] = algo_queries
                all_est_E_0s[algo][T] = algo_est_E_0s
            except Exception as e:
                print(e)
                print('Failed to grab energy estimates for '+algo+'. Try recalculating the results of the algorithm.'); sys.exit(0)

    H = parameters['Hamiltonian']
    real_E_0 = parameters['real_E_0']
    E,vecs = eigh(H)
    # real_E_0 = E[0]
    vecs = [vecs[:,i] for i in range(len(vecs))]
    sv = parameters['sv']

    # # check lengths of data
    # if contains_linear:
    #     if reruns > len(all_exp_vals):
    #         print('Number of linear time series is too small. Reducing reruns.')
    #         reruns=len(all_exp_vals)
    # for algo in parameters['algorithms']:
    #     if reruns > len(all_est_E_0s[algo]):
    #         print('Number of ground state estimations is too small for '+algo+'. Reducing reruns.')
    #         reruns=len(all_est_E_0s[algo])
    #     if reruns > len(all_observables[algo]):
    #         print('Number of observables is too small for '+algo+'. Reducing reruns.')
    #         reruns=len(all_observables[algo]) 

    # create related graphs
    plt.figure()
    plt.title('Overlap')
    vecs_num = len(vecs)
    if vecs_num > 16: vecs_num = 16
    plt.bar(range(vecs_num),[np.abs(sv@vecs[i])**2 for i in range(vecs_num)], width=.6)
    plt.xticks(range(vecs_num), [f'{i:.5}' for i in E[:vecs_num]], rotation=90)
    plt.xlabel('Energy value of eigenstate', labelpad=10)
    plt.ylabel('Overlap with input state')
    plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Spectrum.pdf', bbox_inches='tight')
    plt.show()

    alpha = 1/reruns
    if contains_linear:
        plt.figure()
        avg_exp_vals = np.zeros(len(all_exp_vals[0]), dtype=complex)
        Dt = linear_ts[1]/linear_ts[2]
        for i in range(reruns):
            exp_vals = all_exp_vals[i]
            avg_exp_vals += exp_vals
            plt.plot([i*Dt for i in range(len(exp_vals))], [i.real for i in exp_vals], alpha = alpha, c = 'orange')
            plt.plot([i*Dt for i in range(len(exp_vals))], [i.imag for i in exp_vals], alpha = alpha, c = 'blue')
        avg_exp_vals /= reruns
        plt.plot([i*Dt for i in range(len(exp_vals))], [i.real for i in avg_exp_vals], c = 'orange', label = 'Real')
        plt.plot([i*Dt for i in range(len(exp_vals))], [i.imag for i in avg_exp_vals], c = 'blue', label = 'Imaginary')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.title('Expectation Value with Dt='+str(Dt))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots=True)+'_Expectation_Value.pdf', bbox_inches='tight')
        plt.show()
        

        plt.figure()
        for i in range(reruns):
            exp_vals = all_exp_vals[i]
            plt.plot(fftshift(fftfreq(len(exp_vals), d=Dt)), abs(fftshift(fft(exp_vals))), c = 'purple', alpha = alpha)
        plt.plot(fftshift(fftfreq(len(avg_exp_vals), d=Dt)), abs(fftshift(fft(avg_exp_vals))), c = 'purple', label = 'FFT')
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Expectation Value with Dt='+str(Dt))
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Fourier_Transform_Expectation_Value.pdf', bbox_inches='tight')
        plt.show()

        if fourier_filtering:
            plt.figure()
            ax = plt.axes(projection='3d') 
            for i in range(reruns):
                ff_exp_vals = fourier_filter_exp_vals(all_exp_vals[i], gamma_range, filters)
                for j in range(len(ff_exp_vals)):
                    single_ff_exp_vals = ff_exp_vals[j]
                    plt.plot(gammas[j], [i*Dt for i in range(len(single_ff_exp_vals))], [i.real for i in single_ff_exp_vals], alpha = alpha, c = 'orange')
                    plt.plot(gammas[j], [i*Dt for i in range(len(single_ff_exp_vals))], [i.imag for i in single_ff_exp_vals], alpha = alpha, c = 'blue')
            plt.ylabel('Time')
            ax.set_zlabel('Expectation Value')
            plt.xlabel('gamma')
            plt.title('Expectation Value with Dt='+str(Dt))
            plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots=True)+'gamma='+str(gamma_range[0])+'-'+str(gamma_range[1])+'filters='+str(filters)+'_Expectation_Value.pdf')
            plt.show()
            
            plt.figure()

            fig, axes = plt.subplots(nrows=filters//5+1, ncols=min(5, filters), figsize=(min(5, filters)*8, (filters//5+1)*5))
            if filters == 1 or len(axes)<5: axes = [axes]
            for i in range(reruns):
                if parameters['algorithms']['FDODMD']['full_observable']: data = all_exp_vals[i]
                else: data = [j.real for j in all_exp_vals[i]]
                ff_exp_vals = fourier_filter_exp_vals(data, gamma_range, filters)
                for j in range(len(ff_exp_vals)):
                    single_ff_exp_vals = ff_exp_vals[j]
                    axes[j//5][j%5].set_title('gamma = '+str(gammas[j]))
                    axes[j//5][j%5].plot(fftshift(fftfreq(len(single_ff_exp_vals), d=Dt)), abs(fftshift(fft(single_ff_exp_vals))))
                    if j//5==0: axes[j//5][j%5].set_ylabel('Amplitute')
                    axes[j//5][j%5].set_xlabel('Frequency')
            fig.suptitle('Fourier Transform of Expectation Value with Dt='+str(Dt))
            fig.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'gamma='+str(gamma_range[0])+'-'+str(gamma_range[1])+'filters='+str(filters)+'_Fourier_Transform_Expectation_Value.pdf', bbox_inches='tight')
            fig.show()
    
    if parameters['algorithms']: # if theres at least one algorithm
        colors = {'QCELS':'red', 'ODMD':'blue', 'FDODMD':'purple', 'ML_QCELS':'orange', 'UVQPE':'limegreen', 'VQPE':'darkolivegreen', 'QMEGS':'hotpink'}
        shapes = {'QCELS':'o', 'ODMD':'^', 'FDODMD':'d', 'ML_QCELS':'X', 'UVQPE':'P', 'VQPE':'*', 'QMEGS':'|'}
        
        longest_query = 0
        for algo in all_queries:
            for T in all_queries[algo]:
                num = all_queries[algo][T][-1]
                if longest_query < num:
                    longest_query = num
    
        plt.figure()
        avg_err = {}
        for algo in parameters['algorithms']:
            color = colors[algo]
            shape = shapes[algo]
            for T in all_queries[algo]:
                queries = all_queries[algo][T]
                errs = []
                for r in range(reruns):
                    est_E_0s = all_est_E_0s[algo][T][r]
                    errs.append([abs(w-real_E_0) for w in est_E_0s])

                avg_err = []
                std_err = []
                for j in range(len(errs[0])):
                    temp = []
                    for i in range(len(errs)):
                        temp.append(errs[i][j])
                    avg_err.append(np.average(temp))
                    std_err.append(np.std(temp))
                plt.plot(queries, avg_err, c = color, marker = shape, label = algo+' T='+str(T))
                if show_std: plt.fill_between(queries, np.array(avg_err)-np.array(std_err), np.array(avg_err)+np.array(std_err), color=color, alpha=0.2)
        plt.plot([0,longest_query], [10**-3, 10**-3], label = 'Chemical Accuracy', c = 'black')
        if max_itr != -1: plt.xlim([0, max_itr])
        plt.title('Convergence Absolute Error in Energy for '+parameters['system'])
        plt.ylabel('Absolute Error')
        plt.xlabel('Total Queries')
        plt.legend()
        plt.yscale('log')
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Abs_Error_Queries.pdf', bbox_inches='tight')
        plt.show()

        plt.figure()
        avg_E_0s = {}
        for algo in parameters['algorithms']:
            color = colors[algo]
            shape = shapes[algo]
            avg_E_0s[algo] = {}
            T = max(all_est_E_0s[algo])
            queries = all_queries[algo][T]
            avg_E_0s[algo][T] = np.zeros(len(all_est_E_0s[algo][T][0]))
            for r in range(reruns):
                avg_E_0s[algo][T] += all_est_E_0s[algo][T][r]
            avg_E_0s[algo][T] /= reruns
            plt.plot(queries, avg_E_0s[algo][T], c = color, marker = shape, label = algo+' T='+str(T))
        
        if show_std:
            for algo in parameters['algorithms']:
                color = colors[algo]
                std_exp_vals = []
                T = max(all_est_E_0s[algo])
                for i in range(len(all_est_E_0s[algo][T][0])):
                    tmp = []
                    for r in range(reruns):
                        tmp.append(all_est_E_0s[algo][T][r][i])
                    std_exp_vals.append(np.std(tmp))
                print(all_queries[algo][T])
                plt.fill_between(all_queries[algo][T], avg_E_0s[algo][T]-std_exp_vals, avg_E_0s[algo][T]+std_exp_vals, color=color, alpha=0.2)

        eigs = np.linalg.eigvals(H)
        eigs = np.sort([(eig.real-parameters['shifting'])*parameters['r_scaling'] for eig in eigs])
        for i in range(len(eigs)):
            if i<3: label = 'E'+str(i)
            else: label = ''
            plt.plot([0,longest_query], [eigs[i],eigs[i]], ':', label=label)
        if max_itr != -1: plt.xlim([0, max_itr])
        plt.xlabel('Total Queries ('+str(parameters['shots'])+' per circuit)')
        plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        # dis = (eigs[2]-eigs[0])/2
        # plt.ylim(eigs[0]-dis, eigs[2]+dis)
        # plt.ylim(eigs[0] - 0.1, eigs[0] + 0.1)
        plt.title('Convergence in Energy for '+parameters['system'])
        plt.ylabel('Eigenvalue')
        plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Convergence_Queries.pdf', bbox_inches='tight')
        plt.show()

        if not parameters['const_obs']:
            plt.figure()
            for algo in parameters['algorithms']:
                color = colors[algo]
                shape = shapes[algo]

                errs = []
                for r in range(reruns):
                    est_E_0s = []
                    for T in all_est_E_0s[algo]:
                        est_E_0s.append(all_est_E_0s[algo][T][r][-1])
                    errs.append([abs(w-real_E_0) for w in est_E_0s])
                
                avg_err = []
                std_err = []
                for j in range(len(errs[0])):
                    temp = []
                    for i in range(len(errs)):
                        temp.append(errs[i][j])
                    avg_err.append(np.average(temp))
                    std_err.append(np.std(temp))

                plt.plot(list(all_est_E_0s[algo].keys()), avg_err, c = color, marker = shape, label = algo)
                if show_std: plt.fill_between(list(all_est_E_0s[algo].keys()), np.array(avg_err)-np.array(std_err), np.array(avg_err)+np.array(std_err), color=color, alpha=0.2)

            plt.plot([0,parameters['max_T']], [10**-3, 10**-3], label = 'Chemical Accuracy', c = 'black')
            if max_itr != -1: plt.xlim([0, max_itr])
            plt.title('Convergence Absolute Error in Energy for '+parameters['system'])
            plt.ylabel('Absolute Error')
            plt.xlabel('Total Evolution Time')
            plt.legend()
            plt.yscale('log')
            plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_Abs_Error_Times.pdf', bbox_inches='tight')
            plt.show()

            # for algo in parameters['algorithms']:
            #     # === Raw Data ===
            #     # Format: (final_time, observable_count, value)
            #     data = []
            #     # for T in all_est_E_0s[algo]:
            #     #     # Average E_0 across reruns
            #     #     avg_E_0 = 0.0
            #     #     for r in range(reruns):
            #     #         avg_E_0 += abs(all_est_E_0s[algo][T][r][-1]-real_E_0)
            #     #     avg_E_0 /= reruns

            #     #     observable = all_observables[algo][T][r][-1]
            #     #     point = (observable, T, avg_E_0)
            #     #     data.append(point)
                
            #     final_times = parameters['final_times']
            #     final_observables = parameters['final_observables']

            #     # observables = [int(i) for i in np.linspace(0,300,11)[1:]]
            #     for obs in final_observables:
            #         for T in parameters['final_times']:
            #             try:
            #                 with open('1-Algorithms/Results/'+algo+'_'+make_filename(parameters, add_shots=True, T=T, obs=obs)+'.pkl', 'rb') as file:
            #                     [algo_observables, algo_est_E_0s] = pickle.load(file)
            #                 avg_E_0 = 0.0
            #                 for r in range(reruns):
            #                     avg_E_0 += abs(algo_est_E_0s[r][-1]-real_E_0)
            #                 avg_E_0 /= reruns
            #                 point = (obs, T, avg_E_0)
            #                 data.append(point)
            #             except Exception as e:
            #                 print(e)
            #                 print('Failed to grab energy estimates for '+algo+' with '+str(obs)+' observables. Try recalculating the results of the algorithm.'); break
                        
            #     # === Extract unique axis values ===
                

            #     # === Create index mappings ===
            #     final_time_idx = {v: i for i, v in enumerate(final_times)}
            #     observable_count_idx = {v: i for i, v in enumerate(final_observables)}

            #     # === Initialize heatmap matrix ===
            #     heatmap = np.full((len(final_times), len(final_observables)), np.nan)
                
            #     # === Fill the heatmap matrix ===
            #     for observables, final_time, value in data:
            #         if value==0: value=1e-16
            #         row = final_time_idx[final_time]
            #         col = observable_count_idx[observables]
            #         heatmap[row][col] = value
            #     heatmap = np.where((heatmap <= 0) | np.isnan(heatmap), 0, heatmap)

            #     # === Plot the heatmap ===
            #     from matplotlib.colors import LogNorm
            #     fig, ax = plt.subplots(figsize=(8, 6))
            #     cax = ax.imshow(heatmap, cmap='coolwarm', aspect='auto', norm=LogNorm(vmin=1, vmax=1e-16))
            #     ax.invert_yaxis()
            #     # === Colorbar ===
            #     cbar = fig.colorbar(cax, ax=ax)
            #     cbar.set_label('Absolute Error')

            #     # === Set ticks and labels ===
            #     # tick_positions = range(-1, max(final_times), 10)[1:]
            #     # final_time_labels = range(0,max(final_times), 10)[1:]
            #     ax.set_yticks(range(len(final_times)))
            #     ax.set_yticklabels(final_times)
            #     ax.set_xticks(range(len(final_observables)))
            #     ax.set_xticklabels(final_observables)
            #     ax.tick_params('x', rotation=90)

            #     ax.set_ylabel('Final Simulation Time')
            #     ax.set_xlabel('Observables')
            #     ax.set_title(algo)

            #     # === Annotate cells with values ===
            #     # for i in range(len(observable_counts)):
            #     #     for j in range(len(final_times)):
            #     #         val = heatmap[i, j]
            #     #         if not np.isnan(val):
            #     #             ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

            #     plt.tight_layout()
            #     plt.savefig('2-Graphing/Graphs/'+make_filename(parameters, add_shots =True)+'_'+algo+'_Heatmap.pdf', bbox_inches='tight')
            #     plt.show()



                # plt.figure()
                # data = np.random.rand(10,10)
                # print(data)
                # plt.imshow(data)
                # cbar = plt.colorbar()
                # cbar.set_label('Absolute Error')
                # plt.title('Super Cool graph')
                # plt.xlabel('Observables')
                # plt.ylabel('Final Simulation Time')
                # times = parameters['final_times']
                # plt.xticks(np.linspace(0,times[-1], 9))
                # print(np.linspace(0,longest_query, 9))
                # # plt.yticks(np.linspace(0,longest_query, 9))
                # plt.show()
    
    isolate_graphs(parameters)

def isolate_graphs(parameters):
    contains_linear = check_contains_linear(parameters['algorithms'])
    
    exit_code = os.system('rm -rf Recent_Graphs')
    assert(exit_code == 0)
    print('Attempting to copy newly generated graphs.')
    try: os.mkdir('Recent_Graphs')
    except: pass
    try:
        filename = make_filename(parameters, add_shots=True)
        graph_types = ['Spectrum']
        if parameters['algorithms']:
            graph_types.append('Abs_Error_Queries')
            graph_types.append('Convergence_Queries')
            if not parameters['const_obs']:
                graph_types.append('Abs_Error_Times')
                # for algo in parameters['algorithms']:
                #     graph_types.append(algo+'_Heatmap')    
        if contains_linear:
            graph_types.append('Expectation_Value')
            graph_types.append('Fourier_Transform_Expectation_Value')
        for graph_type in graph_types:
            exit_code = os.system('cp \'2-Graphing/Graphs/'+filename+'_'+graph_type+'.pdf\' Recent_Graphs/'+graph_type+'.pdf')
            assert(exit_code==0)
            if 'FDODMD' in parameters['algorithms'] and graph_type[-17:] == 'Expectation_Value':
                gamma_range = parameters['algorithms']['FDODMD']['gamma_range']
                filters = parameters['algorithms']['FDODMD']['filter_count']
                fn = make_filename(parameters, add_shots=True)+'gamma='+str(gamma_range[0])+'-'+str(gamma_range[1])+'filters='+str(filters)
                exit_code = os.system('cp \'2-Graphing/Graphs/'+fn+'_'+graph_type+'.pdf\' Recent_Graphs/Fourier_Filtered_'+graph_type+'.pdf')
                assert(exit_code==0)      
        print('Successfully copied newly generated graphs. (', end ='')
        for graph_type in graph_types[:len(graph_types)-1]:
            print('Recent_Graphs/'+graph_type+'.pdf, ', end='') 
        print('Recent_Graphs/'+graph_types[-1]+'.pdf)') 
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