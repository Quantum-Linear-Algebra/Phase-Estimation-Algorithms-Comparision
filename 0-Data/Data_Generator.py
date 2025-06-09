from Data_Generator_Helper import *
from qiskit_ibm_runtime import SamplerV2 as Sampler
import pickle
import sys
sys.path.append('.')
from Parameters import *
from Service import empty


def run(parameters, backend):
    sampler = Sampler(backend)
    if parameters['comp_type'] == 'J': from Service import service
    # save data when using files
    num_timesteps = parameters['num_timesteps']
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        Dt = parameters['Dt']
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Re.qpy"
        if empty(filename):
            print("Creating file for Re Dt =", Dt)
            trans_qcs = transpile_exp_vals(Dt, num_timesteps, backend, parameters)
            with open(filename, "wb") as file:
                qpy.dump(trans_qcs, file)
        else:
            print("File found for Re Dt =", Dt)
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Im.qpy"
        if empty(filename):
            print("Creating file for Im Dt =", Dt)
            trans_qcs = transpile_exp_vals(Dt, num_timesteps, backend, parameters, W = 'Im')
            with open(filename, "wb") as file:
                qpy.dump(trans_qcs, file)
        else:
            print("File found for Im Dt =", Dt)
        print()

    # load/generate exp_vals data
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        trans_qcs = []
        Dt = parameters['Dt']
        print("Loading data from file for Real Hadamard Tests")
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Re.qpy"
        with open(filename, 'rb') as file:
            qcs = qpy.load(file)
            trans_qcs.append(qcs)
        print("Loading data from file for Imaginary Hadamard Tests")
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Im.qpy"
        with open(filename, 'rb') as file:
            qcs = qpy.load(file)
            trans_qcs.append(qcs)
        print()
        trans_qcs = sum(trans_qcs, []) # flatten list
        job_correct_size = False
        jobs_tqcs = [trans_qcs]
        # The circuits are divided as into as little jobs as possible
        while(not job_correct_size):
            jobs = []
            job_correct_size = True
            try:
                for tqcs in jobs_tqcs:
                    jobs.append(sampler.run(tqcs, shots = parameters['shots']))
            except:
                job_correct_size = False
                print("Job too large, splitting in half (max "+str(len(jobs_tqcs[0])//2+1)+" circuits per job)... ")
            temp = []
            for tqcs in jobs_tqcs:
                half = int(len(tqcs)/2)
                temp.append(tqcs[:half])
                temp.append(tqcs[half:])
            jobs_tqcs = temp
        if parameters['comp_type'] == 'H':
            print("Saving Parameters")
            # save variables into file with pickle
            job_ids = ""
            for job in jobs:
                job_ids +=job.job_id()
            with open("0-Data/Jobs/"+job_ids+".pkl", "wb") as file:
                info = [num_timesteps, parameters]
                pickle.dump(info, file)
            print("Sending Job")
        if parameters['comp_type'] == 'S':
            print("Running Circuits.")
        results = []
        for job in jobs:
            for result in job.result():
                results.append(result)
        print("Data recieved.")
        print()
    elif parameters['comp_type'] == 'J':
        job_ids = input("Enter Job ID(s):")
        print("Loading parameter data")
        with open("Jobs/"+str(job_ids)+".pkl", "rb") as file:
            [num_timesteps, parameters] = pickle.load(file)
        results = []
        for i in range(len(job_ids)//20):
            job_id = job_ids[i*20:(i+1)*20]
            print("Loading data from job:", job_id)
            job = service.job(job_id)
            for result in job.result():
                results.append(result)
            print("Loaded data from job:", job_id)
        print()

    create_hamiltonian(parameters)

    # print(len(results))
    # construct exp_vals
    exp_vals = []
    if parameters['comp_type'] == 'C':
        print("Generating Data")
        exp_vals = generate_exp_vals(num_timesteps, parameters)
    elif parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        print("Calculating the expectation values.")
        result = results[0:num_timesteps]
        Res = []
        Ims = []
        for j in range(len(result)):
            raw_data = result[j].data
            cbit = list(raw_data.keys())[0]
            Res.append(calculate_exp_vals(raw_data[cbit].get_counts(), parameters['shots']))
        start = num_timesteps
        result = results[start:(start+num_timesteps)]
        for j in range(len(result)):
            raw_data = result[j].data
            cbit = list(raw_data.keys())[0]
            Ims.append(calculate_exp_vals(raw_data[cbit].get_counts(), parameters['shots']))
        else:
            for j in range(len(result)):
                Ims.append(0)
        for i in range(len(Res)):
            exp_vals.append(complex(Res[i], Ims[i]))

    # save expectation Value
    with open("0-Data/Expectation_Values/"+make_filename(parameters, add_shots=True)+".pkl", "wb") as file:
        pickle.dump(exp_vals, file)

    print("Saved expectation values into file.", "(0-Data/Expectation_Values/"+make_filename(parameters, add_shots=True)+".pkl)")