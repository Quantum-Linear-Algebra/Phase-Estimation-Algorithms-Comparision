from Data_Generator_Helper import *
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
import pickle
import sys
sys.path.append('.')
from Parameters import *
from Service import empty, create_service


def run(parameters, backend):
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        Dt = parameters['Dt']
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Re.qpy"
        if empty(filename):
            print("Creating file for Re Dt =", Dt)
            trans_qcs = transpile_exp_vals(parameters, Dt, backend)
            with open(filename, "wb") as file:
                qpy.dump(trans_qcs, file)
        else:
            print("File found for Re Dt =", Dt)
        filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_Im.qpy"
        if empty(filename):
            print("Creating file for Im Dt =", Dt)
            trans_qcs = transpile_exp_vals(parameters, Dt, backend, W='Im')
            with open(filename, "wb") as file:
                qpy.dump(trans_qcs, file)
        else:
            print("File found for Im Dt =", Dt)
        if 'VQPE' in parameters['algorithms']:
            filename = "0-Data/Transpiled_Circuits/"+make_filename(parameters)+"_VQPE.qpy"
            if empty(filename):
                print("Creating file for VQPE Dt =", Dt)
                trans_qcs = transpile_exp_vals(parameters, Dt, backend, W='Im')
                with open(filename, "wb") as file:
                    qpy.dump(trans_qcs, file)
            else:
                print("File found for VQPE Dt =", Dt)
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
        sampler = Sampler(backend)
        if parameters['comp_type'] == 'H':
            job_correct_size = False
            jobs_tqcs = [trans_qcs]
            # The circuits are divided as into as little jobs as possible
            while(not job_correct_size):
                jobs = []
                job_correct_size = True
                for job_tqcs in jobs_tqcs:
                    print(len(job_tqcs)*parameters['shots'])
                    if len(job_tqcs)*parameters['shots']>=10000000: # shot limit
                        job_correct_size = False
                if job_correct_size:
                    try:
                        for tqcs in jobs_tqcs:
                            jobs.append(sampler.run(tqcs, shots = parameters['shots']))
                    except:
                        job_correct_size = False
                if not job_correct_size:
                    print("Job too large, splitting in half (max "+str(len(jobs_tqcs[0])//2)+" circuits per job)... ")
                    temp = []
                    for tqcs in jobs_tqcs:
                        half = int(len(tqcs)/2)
                        temp.append(tqcs[:half])
                        temp.append(tqcs[half:])
                    jobs_tqcs = temp
            print("Saving Parameters.")
            batch_id = jobs[0].job_id()
            job_ids = [job.job_id() for job in jobs]
            with open("0-Data/Jobs/"+batch_id+".pkl", "wb") as file:
                pickle.dump([parameters, job_ids], file)
            print("Sending Job.")
        if parameters['comp_type'] == 'S':
            print("Running Circuits.")
            jobs = [sampler.run(trans_qcs, shots=parameters['shots'])]
        results = []
        for job in jobs:
            for result in job.result():
                results.append(result)
        print("Data recieved.")
        print()
    elif parameters['comp_type'] == 'J':
        batch_id = input("Enter Batch ID:")
        print("Loading parameter data")
        with open("0-Data/Jobs/"+str(batch_id)+".pkl", "rb") as file:
            [parameters, job_ids] = pickle.load(file)
        results = []
        service = create_service()
        for job_id in job_ids:
            print("Loading data from job:", job_id)
            job = service.job(job_id)
            for result in job.result():
                results.append(result)
            print("Loaded data from job:", job_id)
        print()

    create_hamiltonian(parameters)
    
    exp_vals = []
    if parameters['comp_type'] == 'C':
        print("Generating Data")
        exp_vals = generate_exp_vals(parameters)
    elif parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        num_timesteps = parameters['num_timesteps']
        print("Calculating the expectation values.")
        result = results[0:num_timesteps]
        Res = []
        for j in range(len(result)):
            raw_data = result[j].data
            cbit = list(raw_data.keys())[0]
            Res.append(calculate_exp_vals(raw_data[cbit].get_counts(), parameters['shots']))
        Ims = []
        start = num_timesteps
        result = results[start:(start+num_timesteps)]
        for j in range(len(result)):
            raw_data = result[j].data
            cbit = list(raw_data.keys())[0]
            Ims.append(calculate_exp_vals(raw_data[cbit].get_counts(), parameters['shots']))
        for i in range(len(Res)):
            exp_vals.append(complex(Res[i], Ims[i]))
        if 'VQPE' in parameters['algorithms']:
            VQPE_data = []
            start = num_timesteps
            result = results[start:(start+num_timesteps)]
            for j in range(len(result)):
                raw_data = result[j].data
                cbit = list(raw_data.keys())[0]
                VQPE_data.append(calculate_exp_vals(raw_data[cbit].get_counts(), parameters['shots']))
            with open("0-Data/Expectation_Values/"+make_filename(parameters, add_shots=True)+"_VQPE.pkl", "wb") as file:
                pickle.dump(VQPE_data, file)

    # save expectation Value
    with open("0-Data/Expectation_Values/"+make_filename(parameters, add_shots=True)+".pkl", "wb") as file:
        pickle.dump(exp_vals, file)

    print("Saved expectation values into file.", "(0-Data/Expectation_Values/"+make_filename(parameters, add_shots=True)+".pkl)")

def save_job_ids_params(parameters):
    job_ids = input("Enter Job ID(s):")
    job_ids = [job_ids[i*20:(i+1)*20] for i in range(len(job_ids)//20)]
    print(job_ids)
    with open("0-Data/Jobs/"+job_ids[0]+".pkl", "wb") as file:
        pickle.dump([parameters, job_ids], file)
