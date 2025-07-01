from qiskit_ibm_runtime import SamplerV2 as Sampler
import pickle
import sys
sys.path.append('.')
from Service import empty, create_service
from Parameters import make_filename
sys.path.append('./1-Algorithms/Algorithms')

import subprocess, os, numpy as np
from scipy.linalg import expm, eigh, svd
from scipy.fft import fft, ifft

from qiskit import transpile
from qiskit import qpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import UnitaryGate

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper

# Prevent annoying migration warnings
import warnings 
warnings.simplefilter("ignore")

def closest_unitary(A):
    """ 
    Description: Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A. Used when qiskit fails to transpile
    unitary gate due to float point rounding.

    Args: Unitary matrix which qiskit fails to diagonalize: A

    Return: Unitary as an np matrix
    """
    V, __, Wh = svd(A)
    U = np.matrix(V.dot(Wh))
    return U

def create_hamiltonian(parameters, scale=True, show_steps=False):
    '''
    Create a system hamiltonian for the following systems:
     - Tranverse Field Ising Model (TFI)
     - Heisenberg Spin Model (SPI)
     - Hubbard Model (HUB)
     - Dihydrogen (H_2)

    Parameters:
     - parameters: a dictionary of parameters for contructing
       the Hamiltonian containing the following information
        - system: the system written as either: TFI, SPI, HUB, H_2
        - sites: the number of sites, default is 2
        - scaling: scales the eigenvalues to be at most this number
        - shifting: shift the eigenvalues by this value
        system specific parameters:
        TFI
        - g: magnetic field strength
        SPI
        - J: coupling strength
        HUB
        - t: left-right hopping
        - U: up-down hopping
        - x: x size of latice
        - y: y size of latice
        H_2
        - distance: the distance between two Hydrogen
     - show_steps: if true then debugging print statements
                   are shown
    
    Effects:
       This method also creates parameter['r_scaling'] which
       is used for recovering the original energy after ODMD.
     
    Returns:
     - H: the created hamiltonian
     - real_H_0: the minimum energy of the unscaled system
    '''

    system = parameters['system']
    scale_factor = parameters['scaling']
    shifting = parameters['shifting']
    if 'sites' in parameters.keys(): qubits = parameters['sites']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if system == "TFI":
        g = parameters['g']
        # construct the Hamiltonian
        # with Pauli Operators in Qiskit ^ represents a tensor product
        if show_steps: print("H = ", end='')
        for i in range(qubits-1):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i or j == i+1):
                    temp ^= Pauli('Z')
                else:
                    temp ^= Pauli('I')
            H += -temp.to_matrix()
            if show_steps: print("-"+str(temp)+" ", end='')
        # peroidic bound
        temp = Pauli('')
        for j in range(qubits):
            if (j == 0 or j == qubits-1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -temp.to_matrix()
        if show_steps: print("-"+str(temp)+" ", end='')
        for i in range(qubits):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i):
                    temp ^= Pauli('X')
                else:
                    temp ^= Pauli('I')
            H += -g*temp.to_matrix()
            if show_steps: print("-"+str(g)+"*"+str(temp)+" ", end='')
        if show_steps: print("\n")
    elif system == "SPI":
        qubits = parameters['sites']
        J = parameters['J']
        def S(index, coupling):
            temp = Pauli('')
            for j in range(qubits):
                if j == index:
                    temp ^= Pauli(coupling)
                else:
                    temp ^= Pauli('I')
            return 1/2*temp.to_matrix()
        if show_steps: print("H = ", end='\n')
        for qubit in range(qubits-1):
            H += S(qubit, 'X')@S(qubit+1, 'X')
            H += S(qubit, 'Y')@S(qubit+1, 'Y')
            H += S(qubit, 'Z')@S(qubit+1, 'Z')
        H += S(qubits-1, 'X')@S(0, 'X')
        H += S(qubits-1, 'Y')@S(0, 'Y')
        H += S(qubits-1, 'Z')@S(0, 'Z')
        H *= J
        if show_steps: print(H)
    elif system == "HUB":
        qubits = parameters['sites']
        x = parameters['x']
        y = parameters['y']
        U = parameters['U']
        t = parameters['t']       
        # coupling portion
        Sd = np.array([[0,0],[1,0]])
        S = np.array([[0,1],[0,0]])
        I = np.eye(2)
        left_right_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for op in [Sd]:
            for site in range(qubits):
                curr_x = site%x
                curr_y = site//x%y
                # couple sites in square latice
                neighbors = []
                if curr_x != 0:   neighbors.append((site-1)%qubits)
                if curr_x != x-1: neighbors.append((site+1)%qubits)
                if curr_y != 0:   neighbors.append((site+x)%qubits)
                if curr_y != y-1: neighbors.append((site-x)%qubits)
                for neighbor in neighbors:
                    temp = [1]
                    for site_ in range(qubits):
                        if site_ == site: temp = np.kron(temp, op)
                        elif site_ == neighbor: temp = np.kron(temp, op.T)
                        else: temp = np.kron(temp, I)
                    left_right_hopping_term+=temp
        left_right_hopping_term *=-t
        # number operator portion
        op1 = np.kron(Sd, Sd)
        op2 = np.kron(S, S)
        num = op1@op2
        up_down_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for place in range(qubits-1):
            temp = [1]
            for index in range(qubits-1):
                if index == place: temp = np.kron(temp, num) 
                else: temp = np.kron(temp, I) 
            up_down_hopping_term+=temp
        up_down_hopping_term*=U

        H = up_down_hopping_term+left_right_hopping_term
    elif system == 'H_2':
        distance = parameters['distance']
        parameters['sites'] = 1
        driver = PySCFDriver(
            atom=f'H .0 .0 .0; H .0 .0 {distance}',
            basis='sto3g'
        )
        molecule = driver.run()
        mapper = ParityMapper()
        fer_op = molecule.hamiltonian.second_q_op()
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        H = tapered_mapper.map(fer_op)
        H = H.to_matrix()

    val, vec = eigh(H)
    real_E_0 = val[0]

    if scale:
        if show_steps:
            print("Original eigenvalues:", val)
            print("Original eigenvectors:\n", vec)
            print("Original Matrix:")
            for i in H:
                for j in i:
                    print(j, end = '\t\t')
                print()
        # # calculate the max magnitute eigenvalue (not correct yet)
        # max_iter = 1000
        # tol = 10**-10
        # n = H.shape[0]
        # v = np.random.rand(n) + 1j * np.random.rand(n)  # Initial random complex vector
        # lambda_old = 0
        # for _ in range(max_iter):
        #     v_next = H @ v
        #     print(v)
        #     lambda_new = np.vdot(v_next, v) / np.vdot(v, v)  # Rayleigh quotient
        #     if abs(lambda_new - lambda_old) < tol: break
        #     v = v_next
        #     lambda_old = lambda_new
        # scale eigenvalues of the Hamiltonian
        n = 2**qubits
        largest_eigenvalue = np.max(abs(val)) # use lambda_new when the above code segment
        if show_steps: print("Largest Eigenvalue =", largest_eigenvalue)
        parameters["r_scaling"] = largest_eigenvalue/scale_factor
        H *= scale_factor/largest_eigenvalue
        H += shifting*np.eye(n)
        if show_steps:
            val, vec = eigh(H)
            print("Scaled eigenvalues:", val)
            print("Scaled eigenvectors:\n", vec)
            min_eigenvalue = np.min(val)
            print("Lowest energy eigenvalue", min_eigenvalue); print()
    return H, real_E_0

def run_hadamard_test(controlled_U, statevector, W = 'Re', shots=100):
    '''
    Run a transpiled hadamard test quantum circuit.

    Parameters:
     - controlled_U: the control operation to check phase of
     - statevector: a vector to initalize the statevector of
                    eigenqubits
     - W: what type of hadamard test to use (Re or Im)
     - shots: the number of shots to run the test with 

    Returns:
     - re: the real part of expection value measured
    '''

    aer_sim = AerSimulator(noise_model=NoiseModel())
    trans_qc = create_hadamard_test(aer_sim, controlled_U, statevector, W = W)
    counts = aer_sim.run(trans_qc, shots = shots).result().get_counts()
    exp_val = calculate_exp_vals(counts, shots)
    return exp_val

def create_hadamard_test(backend, controlled_U, statevector, W = 'Re'):
    '''
    Creates a transpiled hadamard test for the specificed backend.

    Parameters:
     - backend: the backend to transpile the circuit on
     - controlled_U: the control operation to check phase of
     - statevector: a vector to initalize the statevector of
                    eigenqubits
     - W: what type of hadamard test to use (Re or Im)
    
    Returns:
     - trans_qc: the transpiled circuit
    '''
    
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(controlled_U.num_qubits-1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)
    # qc.h(qr_eigenstate)
    qc.initialize(statevector, qr_eigenstate)
    qc.append(controlled_U, qargs = [qr_ancilla[:]] + qr_eigenstate[:] )
    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    trans_qc = transpile(qc, backend, optimization_level=3)
    return trans_qc

def calculate_exp_vals(counts, shots):
    '''
    Calculates the real or imaginary of the expectation
    value depending on if the counts provided are from
    the real or the imaginary Hadamard test.

    Parameters:
     - counts: the count object returned from result
     - shots: the number of shots to run the test with 

    Returns:
     - meas: the desired expection value
    '''
    p0 = 0
    if counts.get('0') is not None:
        p0 = counts['0']/shots
    meas = 2*p0-1
    return meas

def make_overlap(ground_state, p):
    '''
    Creates a statevector with p probability of overlap with the ground state.

    Parameters:
     - groundstate: the ground statevector
     - p: the desired probability of overlap with the groundstate
    
    Returns:
     - phi: a statevector with the conditions
    '''
    length = len(ground_state)
    # generate a random vector with no overlap with the ground state
    random_vec = np.random.randn(length) + 1j * np.random.randn(length)
    random_vec -= np.vdot(ground_state, random_vec) * ground_state
    random_vec /= np.linalg.norm(random_vec)
    # contruct a statevector with p prob of overlap with groundstate
    phi = np.sqrt(p) * ground_state + np.sqrt(1 - p) * random_vec
    return phi

def hadamard_test_circuit_info(Dt, parameters, ML_QCELS=False, pauli_string=''):
    '''
    Gets information for creating exp_vals circuits. Creates controlled unitaries,
    and initialization statevector.

    Parameters:
     - Dt: the time step
     - parameters: the parmeters for the
                   hamiltonian contruction
    
    Returns:
     - gates: the controlled unitary gates
     - statevector: the initialization state vector
    '''
    statevector = parameters['sv']
    VQPE = pauli_string!=''
    use_F3C = not VQPE and parameters['system'] == 'TFI' and parameters['method_for_model']=="F"
    
    num_timesteps = int(parameters['observables']/2)
    if parameters['const_obs'] and (VQPE or parameters['algorithms'] == ['VQPE']): num_timesteps = int(num_timesteps/(len(parameters['pauli_strings'])+1))

    if ML_QCELS:
        time_steps = set()
        iteration = 0
        time_steps_per_itr = parameters['ML_QCELS_time_steps']
        while len(time_steps) < num_timesteps:
            for i in range(time_steps_per_itr):
                time = 2**iteration*i
                if time in time_steps: continue
                time_steps.add(time)
            iteration+=1
        time_steps = np.sort(list(time_steps))
    
    gates = []
    if use_F3C:
        coupling = 1
        scaling = parameters['scaling']
        sites = parameters['sites']
        g = parameters['g']
        trotter = parameters['trotter']
        if ML_QCELS:
            for time_step in time_steps:
                gates.append(generate_TFIM_gates(sites, 1, time_step*Dt, g, scaling, coupling, trotter, '../f3cpp', include_0 = False)[0])
        else:
            gates = generate_TFIM_gates(sites, num_timesteps, Dt, g, scaling, coupling, trotter, '../f3cpp')
    else:
        ham,_ = create_hamiltonian(parameters)
        if ML_QCELS:
            for i in time_steps:
                exp_ham = -1j*ham*i*Dt
                max_mag = 0
                for j in range(len(exp_ham)):
                    for k in range(len(exp_ham[j])):
                        mag = abs(exp_ham[j][k])
                        if max_mag < mag:
                            max_mag = mag
                if max_mag > 10**17:
                    scale = 10**17/max_mag
                    for j in range(len(exp_ham)):
                        for k in range(len(exp_ham[j])):
                            exp_ham[j][k] *= scale
                mat = expm(exp_ham)
                controlled_U = UnitaryGate(closest_unitary(mat)).control(annotated="yes")
                gates.append(controlled_U)
        else:
            gates = []
            for i in range(num_timesteps):
                mat = expm(-1j*ham*Dt*i)
                if VQPE:
                    pauli = Pauli(pauli_string).to_matrix()
                    mat = pauli@mat
                controlled_U = UnitaryGate(closest_unitary(mat)).control(annotated="yes")
                gates.append(controlled_U)
    return gates, statevector

def generate_exp_vals(parameters, reruns):
    '''
    Generate the exp_vals spectrum

    Parameters:
     - parameters: the parmeters for the
                   hamiltonian contruction

    Returns:
     - exp_vals: the data generated
    '''

    Dt = parameters['Dt']
    observables = parameters['observables']
    num_timesteps = int(observables/2)
    sv = parameters['sv']
    H,_ = create_hamiltonian(parameters)
    E, vecs = eigh(H)
    spectrum = []
    for i in range(len(vecs)):
        spectrum.append(np.abs(sv.conj().T@vecs[:,i])**2)
    
    all_exp_vals = {}
    if not(parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']):
        all_exp_vals['linear'] = []
    if parameters['const_obs'] and 'ML_QCELS' in parameters['algorithms']:
        all_exp_vals['sparse'] = []
    if 'VQPE' in parameters['algorithms']:
        all_exp_vals['vqpets'] = []
    
    for _ in  range(reruns):
        if 'linear' in all_exp_vals:
            exp_vals = []
            for i in range(num_timesteps):
                exp_vals.append(np.sum(np.array(spectrum)*np.exp(-1j*E*i*Dt)))
            all_exp_vals['linear'].append(exp_vals)
        if 'sparse' in all_exp_vals:
            exp_vals = {}
            iteration = 0
            time_steps_per_itr = parameters['ML_QCELS_time_steps']
            while len(exp_vals) < num_timesteps:
                for i in range(time_steps_per_itr):
                    time = 2**iteration*i
                    if time in exp_vals: continue
                    exp_vals[time] = np.sum(np.array(spectrum)*np.exp(-1j*E*time*Dt))
                iteration+=1
            all_exp_vals['sparse'].append(exp_vals)
        if 'vqpets' in all_exp_vals:
            exp_vals = []
            length = num_timesteps
            if parameters['const_obs']: length = int(num_timesteps/((len(parameters['pauli_strings'])+1)))
            for i in range(length):
                exp_vals.append(np.sum(np.array(spectrum)*E*np.exp(-1j*E*i*Dt)))
            all_exp_vals['vqpets'].append(exp_vals)
    return all_exp_vals

def transpile_hadamard_tests(parameters, Dt, backend, W='Re', ML_QCELS=False, pauli_string=''):
    '''
    Transpile the related hadamard tests to generate exp_vals

    Parameters:
     - Dt: the time step
     - backend: the backend to transpile on
     - parameters: the parmeters for the hamiltonian
                   contruction

    Returns:
     - trans_qcs: the transpiled circuits
    '''

    tqcs = []
    gates, statevector = hadamard_test_circuit_info(Dt, parameters, ML_QCELS=ML_QCELS, pauli_string=pauli_string)
    for controlled_U in gates:
        tqcs.append(create_hadamard_test(backend, controlled_U, statevector, W=W))
    return tqcs

def generate_TFIM_gates(qubits, steps, dt, g, scaling, coupling, trotter, location, include_0 = True):
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    
    # calculate new scaled parameters
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    for i in range(qubits-1):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i or j == i+1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -temp.to_matrix()
    for i in range(qubits):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i):
                temp ^= Pauli('X')
            else:
                temp ^= Pauli('I')
        H += -g*temp.to_matrix()

    largest_eig = np.max(abs(np.linalg.eigvals(H)))
    coupling *= scaling/largest_eig
    g *= scaling/largest_eig

    gates = []
    if not os.path.exists("TFIM_Operators"):
        os.mkdir("TFIM_Operators")
    
    # add timestep where dt = 0
    if include_0:
        with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
            f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
            f.write("[Trotter]\nsteps = 1\ndt = 0\n\n") # maybe need new number for steps
            f.write("[Jy]\nvalue = 0\n\n")
            f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
            f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
            f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = 2\nstep = 1\n")
        exe = location+"/release/examples/f3c_time_evolution_TFYZ"
        with open("TFIM_Operators/garbage.txt", "w") as file:
            subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"], stdout=file)
        os.remove("TFIM_Operators/garbage.txt")
        os.remove("TFIM_Operators/Operator_Generator.ini")
        qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
        gate = qc.to_gate(label = "TFIM 0").control()
        gates.append(gate)
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
        steps -= 1

    steps *= trotter
    dt    /= trotter
    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = "+str(steps)+"\ndt = "+str(dt)+"\n\n") # maybe need new number for steps
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = "+str(steps+1)+"\nstep = 1\n")
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    with open("TFIM_Operators/garbage.txt", "w") as file:
        subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"], stdout=file)
    os.remove("TFIM_Operators/garbage.txt")
    os.remove("TFIM_Operators/Operator_Generator.ini")
    for step in range(1, steps+1):
        if step % trotter == 0:
            qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
            gate = qc.to_gate(label = "TFIM "+str(step)).control()
            gates.append(gate)
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
    os.rmdir("TFIM_Operators")
    return gates

def run(parameters, returns):
    backend = returns['backend']
    reruns = parameters['reruns']
    if parameters['comp_type'] == 'J': job_ids = returns['job_ids']

    used_time_series = []
    if not (parameters['const_obs'] and parameters['algorithms'] == ['ML_QCELS']): used_time_series.append('linear')
    if parameters['const_obs'] and 'ML_QCELS' in parameters['algorithms']: used_time_series.append('sparse')
    if 'VQPE' in parameters['algorithms']: used_time_series.append('vqpets')
    
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        try: os.mkdir('0-Data/Transpiled_Circuits')
        except: pass

        Dt = parameters['Dt']
        if 'linear' in used_time_series:
            filename = '0-Data/Transpiled_Circuits/linear_'+make_filename(parameters)+'_Re.qpy'
            if empty(filename):
                print('Creating file for Linear Real Hadamard test with Dt =', Dt)
                trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Re')
                with open(filename, 'wb') as file:
                    qpy.dump(trans_qcs, file)
            else:
                print('File found for Linear Real Hadamard test with Dt =', Dt)
            filename = '0-Data/Transpiled_Circuits/linear_'+make_filename(parameters)+'_Im.qpy'
            if empty(filename):
                print('Creating file for Linear Imaginary Hadamard test with Dt =', Dt)
                trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Im')
                with open(filename, 'wb') as file:
                    qpy.dump(trans_qcs, file)
            else:
                print('File found for Linear Imaginary Hadamard test with Dt =', Dt)      
        if 'sparse' in used_time_series:
            filename = '0-Data/Transpiled_Circuits/sparse_'+make_filename(parameters)+'_Re.qpy'
            if empty(filename):
                print('Creating file for Sparse Real Hadamard test with Dt =', Dt)
                trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Re', ML_QCELS=True)
                with open(filename, 'wb') as file:
                    qpy.dump(trans_qcs, file)
            else:
                print('File found for Sparse Real Hadamard test with Dt =', Dt)
            filename = '0-Data/Transpiled_Circuits/sparse_'+make_filename(parameters)+'_Im.qpy'
            if empty(filename):
                print('Creating file for Sparse Imaginary Hadamard test with Dt =', Dt)
                trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Im', ML_QCELS=True)
                with open(filename, 'wb') as file:
                    qpy.dump(trans_qcs, file)
            else:
                print('File found for Sparse Imaginary Hadamard test with Dt =', Dt)      
        if 'vqpets' in used_time_series:
            pauli_strings = parameters['pauli_strings']
            for pauli_string in pauli_strings.paulis:
                pauli_string = str(pauli_string)
                filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+make_filename(parameters)+'_Re.qpy'
                if empty(filename):
                    print('Creating file for '+pauli_string+' Real Hadamard test with Dt =', Dt)
                    trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Real', pauli_string=pauli_string)
                    with open(filename, 'wb') as file:
                        qpy.dump(trans_qcs, file)
                else:
                    print('File found for '+pauli_string+' Imaginary Hadamard test with Dt =', Dt)
                filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+make_filename(parameters)+'_Im.qpy'
                if empty(filename):
                    print('Creating file for '+pauli_string+' Imaginary Hadamard test with Dt =', Dt)
                    trans_qcs = transpile_hadamard_tests(parameters, Dt, backend, W='Im', pauli_string=pauli_string)
                    with open(filename, 'wb') as file:
                        qpy.dump(trans_qcs, file)
                else:
                    print('File found for '+pauli_string+' Imaginary Hadamard test with Dt =', Dt)      
        print()

    # load/generate exp_vals data
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        trans_qcs = []
        Dt = parameters['Dt']
        for run in range(reruns):
            print('Run', run+1)
            if 'linear' in used_time_series:
                print('  Loading data from file for Linear Real Hadamard Tests.')
                filename = '0-Data/Transpiled_Circuits/linear_'+make_filename(parameters)+'_Re.qpy'
                with open(filename, 'rb') as file:
                    qcs = qpy.load(file)
                    trans_qcs.append(qcs)
                print('  Loading data from file for Linear Imaginary Hadamard Tests.')
                filename = '0-Data/Transpiled_Circuits/linear_'+make_filename(parameters)+'_Im.qpy'
                with open(filename, 'rb') as file:
                    qcs = qpy.load(file)
                    trans_qcs.append(qcs)
            if 'sparse' in used_time_series:
                print('  Loading data from file for Sparse Real Hadamard Tests.')
                filename = '0-Data/Transpiled_Circuits/sparse_'+make_filename(parameters)+'_Re.qpy'
                with open(filename, 'rb') as file:
                    qcs = qpy.load(file)
                    trans_qcs.append(qcs)
                print('  Loading data from file for Sparse Imaginary Hadamard Tests.')
                filename = '0-Data/Transpiled_Circuits/sparse_'+make_filename(parameters)+'_Im.qpy'
                with open(filename, 'rb') as file:
                    qcs = qpy.load(file)
                    trans_qcs.append(qcs)
            if 'vqpets' in used_time_series:
                for pauli_string in pauli_strings.paulis:
                    pauli_string = str(pauli_string)
                    filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+make_filename(parameters)+'_Re.qpy'
                    print('  Loading data from file for '+pauli_string+' Real Hadamard Tests.')
                    with open(filename, 'rb') as file:
                        qcs = qpy.load(file)
                        trans_qcs.append(qcs)
                    filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+make_filename(parameters)+'_Im.qpy'
                    print('  Loading data from file for '+pauli_string+' Imaginary Hadamard Tests.')
                    with open(filename, 'rb') as file:
                        qcs = qpy.load(file)
                        trans_qcs.append(qcs)
            
        print()
        trans_qcs = sum(trans_qcs, []) # flatten list
        sampler = Sampler(backend)
        if parameters['comp_type'] == 'H':
            job_correct_size = False
            jobs_tqcs = [trans_qcs]
            # Divide the circuits into multiple smaller jobs
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
                    print('Job too large, splitting in half (max '+str(len(jobs_tqcs[0])//2)+' circuits per job)... ')
                    temp = []
                    for tqcs in jobs_tqcs:
                        half = int(len(tqcs)/2)
                        temp.append(tqcs[:half])
                        temp.append(tqcs[half:])
                    jobs_tqcs = temp
            print('Saving Parameters.')
            batch_id = jobs[0].job_id()
            job_ids = [job.job_id() for job in jobs]
            try: os.mkdir('0-Data/Jobs')
            except: pass
            with open('0-Data/Jobs/'+batch_id+'.pkl', 'wb') as file:
                pickle.dump([parameters, job_ids], file)
            print('Sending Job.')
        if parameters['comp_type'] == 'S':
            print('Running Circuits.')
            jobs = [sampler.run(trans_qcs, shots=parameters['shots'])]
        results = []
        for job in jobs:
            for result in job.result():
                results.append(result)
        print('Data recieved.')
        print()
    elif parameters['comp_type'] == 'J':
        results = []
        service = create_service()
        for job_id in job_ids:
            print('Loading data from job:', job_id)
            job = service.job(job_id)
            for result in job.result():
                results.append(result)
            print('Loaded data from job:', job_id)
        create_hamiltonian(parameters)
        print()
    
    all_exp_vals = {}
    if parameters['comp_type'] == 'C':
        print('Generating Data')
        all_exp_vals = generate_exp_vals(parameters, reruns)
    elif parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        if 'linear' in used_time_series: all_exp_vals['linear'] = []
        if 'sparse' in used_time_series: all_exp_vals['sparse'] = []
        if 'vqpets' in used_time_series: all_exp_vals['vqpets'] = []
        observables = parameters['observables']
        num_timesteps = int(observables/2)
        shots = parameters['shots']
        for r in range(reruns):
            print('Run', r+1)
            print('  Calculating the expectation values from circuit data.')
            for i in range(len(used_time_series)):
                if 'VQPE' in parameters['algorithms']:
                    if parameters['const_obs']:
                        if parameters['algorithms'] == ['VQPE']:
                            index = i*observables//((len(pauli_strings)+1))+r*observables
                        else:
                            # last term adjusts the index to account for the fact that the VQPE time series data is slightly shorter
                            index = i*observables + r*((len(used_time_series)-1)*observables + observables//((len(pauli_strings)+1))*len(pauli_strings))
                    else: index = (i+r*(len(used_time_series)+len(pauli_strings)-1))*observables
                else: index = (i+r*len(used_time_series))*observables
                if used_time_series[i] == 'sparse':
                    list_exp_vals = calc_all_exp_vals(results[index:index+observables], shots)
                    time_steps = set()
                    iteration = 0
                    time_steps_per_itr = parameters['ML_QCELS_time_steps']
                    while len(time_steps) < num_timesteps:
                        for j in range(time_steps_per_itr):
                            time = 2**iteration*j
                            if time in time_steps: continue
                            time_steps.add(time)
                        iteration+=1
                    time_steps = np.sort(list(time_steps))
                    exp_vals = {}
                    for j in range(len(time_steps)):
                        exp_vals[time_steps[j]] = list_exp_vals[j]
                    all_exp_vals['sparse'].append(exp_vals)
                elif used_time_series[i] == 'vqpets':
                    pauli_strings = parameters['pauli_strings']
                    if parameters['const_obs']: vqpe_obs = observables//((len(pauli_strings)+1))
                    else: vqpe_obs = observables
                    Hexp_vals = np.zeros(vqpe_obs//2, dtype=complex)
                    for p in range(len(pauli_strings)):
                        start = index+p*vqpe_obs
                        pauli_string = pauli_strings.paulis[p]
                        coeff = pauli_strings.coeffs[p]
                        exp_vals = calc_all_exp_vals(results[start:start+vqpe_obs], shots)
                        Hexp_vals += [k*coeff for k in exp_vals]
                    all_exp_vals['vqpets'].append(Hexp_vals)
                elif parameters['algorithms'] == ['VQPE'] and parameters['const_obs']:
                    all_exp_vals[used_time_series[i]].append(calc_all_exp_vals(results[index:index+observables//(len(pauli_strings)+1)], shots))
                else: all_exp_vals[used_time_series[i]].append(calc_all_exp_vals(results[index:index+observables], shots))

    # save expectation values
    try: os.mkdir('0-Data/Expectation_Values')
    except: pass
    for key in all_exp_vals.keys():
        filename = '0-Data/Expectation_Values/'+key+'_'+make_filename(parameters, add_shots=True)+'.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(all_exp_vals[key], file)
        print('Saved expectation values into file.', '('+filename+')')  
    # fourier filtering
    if parameters['fourier_filtering']:
        print('\nDenoising expectation values.')
        gamma_range = parameters['gamma_range']
        filter_count = parameters['filter_count']
        gammas = np.linspace(gamma_range[0], gamma_range[1], filter_count)
        fourier_all_exp_vals = {}
        for key in all_exp_vals:
            if key == 'sparse':
                pass
            else:
                fourier_all_exp_vals[key] = []
                for exp_vals in all_exp_vals[key]:
                    filtered_exp_vals = []
                    fft_exp_vals = fft(exp_vals)
                    fft_median = np.median(fft_exp_vals)
                    for gamma in gammas:
                        new_exp_vals = ifft([i*(i>gamma*fft_median) for i in fft_exp_vals])
                        filtered_exp_vals.append(new_exp_vals)
                    fourier_all_exp_vals[key].append(filtered_exp_vals)
        try: os.mkdir('0-Data/Expectation_Values/Denoised')
        except: pass
        for key in fourier_all_exp_vals:
            filename = '0-Data/Expectation_Values/Denoised/'+key+'_'+make_filename(parameters, fourier_filtered=True, add_shots=True)+'.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(fourier_all_exp_vals[key], file)
            print('Saved fourier filtered expectation values into file.', '('+filename+')')
    

def calc_all_exp_vals(results, shots):
    num_timesteps = int(len(results)/2) 
    result = results[0:num_timesteps]
    Res = []
    for j in range(len(result)):
        raw_data = result[j].data
        cbit = list(raw_data.keys())[0]
        Res.append(calculate_exp_vals(raw_data[cbit].get_counts(), shots))
    Ims = []
    start = num_timesteps
    result = results[start:(start+num_timesteps)]
    for j in range(len(result)):
        raw_data = result[j].data
        cbit = list(raw_data.keys())[0]
        Ims.append(calculate_exp_vals(raw_data[cbit].get_counts(), shots))
    exp_vals = []
    for i in range(len(Res)):
        exp_vals.append(complex(Res[i], Ims[i]))
    return exp_vals

def save_job_ids_params(parameters):
    job_ids = input('Enter Job ID(s):')
    job_ids = [job_ids[i*20:(i+1)*20] for i in range(len(job_ids)//20)]
    print(job_ids)
    with open('0-Data/Jobs/'+job_ids[0]+'.pkl', 'wb') as file:
        pickle.dump([parameters, job_ids], file)


if __name__ == '__main__':
    from Comparison import parameters
    from Parameters import check
    returns = check(parameters)
    run(parameters, returns)
    # data.save_job_ids_params(parameters)
