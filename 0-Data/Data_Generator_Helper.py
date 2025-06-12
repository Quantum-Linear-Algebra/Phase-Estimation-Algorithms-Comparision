import subprocess, os, numpy as np
from scipy.linalg import expm, eigh

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
        mapper = ParityMapper(num_sites=molecule.num_sites)
        fer_op = molecule.hamiltonian.second_q_op()
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        H = tapered_mapper.map(fer_op)
        H = H.to_matrix()

    if scale:
        val, vec = eigh(H)
        real_E_0 = val[0]
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

def hadamard_test(controlled_U, statevector, W = 'Re', shots=100):
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
    re = calculate_exp_vals(counts, shots)
    return re

def create_hadamard_test(backend, controlled_U, statevector, W = 'Re'):
    '''
    Creates a transpiled hadamardd test for the specificed backend.

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

def exp_vals_circuit_info(Dt, parameters):
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
    ham,_ = create_hamiltonian(parameters)
    _,vec = eigh(ham)
    ground_state = vec[:,0]
    if 'overlap' in parameters: statevector = make_overlap(ground_state, parameters['overlap'])
    else: statevector = ground_state
    use_F3C = parameters['system']=="TFI" and parameters['method_for_model']=="F"
    
    gates = []
    if use_F3C:
        coupling = 1
        if 'scaling' in parameters: scaling = parameters['scaling']
        else: scaling = 1
        gates = generate_TFIM_gates(parameters['sites'], parameters['num_timesteps'], Dt, parameters['g'], scaling, coupling, parameters['trotter'], '../f3cpp')
    else:
        for i in range(parameters['num_timesteps']):
            mat = expm(-1j*ham*Dt*i)
            controlled_U = UnitaryGate(mat).control(annotated="yes")
            gates.append(controlled_U)
    return gates, statevector

def generate_exp_vals(parameters):
    '''
    Generate the exp_vals spectrum

    Parameters:
     - parameters: the parmeters for the
                   hamiltonian contruction

    Returns:
     - exp_vals: the data generated
    '''

    Dt = parameters['Dt']
    H,_ = create_hamiltonian(parameters)
    E, vecs = eigh(H)
    ground_state = vecs[:,0]
    sv = make_overlap(ground_state, parameters['overlap'])
    exp_vals = []
    for i in range(parameters['num_timesteps']):
        exp_vals.append(np.sum(np.exp(-1j*E*i*Dt)*np.abs(sv)**2))
    return exp_vals

def transpile_exp_vals(parameters, Dt, backend, W='Re'):
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

    trans_qcs = []
    gates, statevector = exp_vals_circuit_info(Dt, parameters)
    for controlled_U in gates:
        trans_qcs.append(create_hadamard_test(backend, controlled_U, statevector, W=W))
    return trans_qcs

def transpile_Hexp_vals(parameters, Dt, backend):
    trans_qcs =[]
    return trans_qcs

def generate_TFIM_gates(qubits, steps, dt, g, scaling, coupling, trotter, location):
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