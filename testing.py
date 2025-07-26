import numpy as np
pi = np.pi 
from Service import create_hardware_backend

from scipy.linalg import expm, eigh, norm

from qiskit import transpile
from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation

from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService as QRS

def create_hadamard_tests(parameters, backend, U:UnitaryGate, statevector=[], W = 'Re', modified=True):
    '''
    Creates a transpiled hadamard tests for the specificed backend.

    Parameters:
     - backend: the backend to transpile the circuit on
     - controlled_U: the control operation to check phase of
     - statevector: a vector to initalize the statevector of
                    eigenqubits
     - W: what type of hadamard tests to use (Re or Im)
     - modified: uses the modified hadamard test if true
    
    Returns:
     - trans_qc: the transpiled circuit
    '''
    qubits = parameters['sites']
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)
    if modified:
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.ch(qr_ancilla,qr_eigenstate[0])
                for qubit in range(1, qubits):
                    qc_init.cx(qubit, qubit+1)
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.ch(qr_ancilla, qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate.control(annotated="yes"))
        
        qc = qc.compose(qc_init)
        qc = qc.compose(U, range(1, qubits+1))
        qc.x(0)
        qc = qc.compose(qc_init)
        qc.x(0)

        sv = np.zeros(2**qubits)
        sv[0]=1
        ev = complex(U.to_matrix()[0][0])
        print(ev)
        phase = np.log(ev)
        print(phase)
        phase = phase.imag
        print(phase)
        qc.rz(phase, qr_ancilla)
    else:
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.ch(qr_eigenstate[0])
                for qubit in range(qubits):
                    qc_init.x(qr_eigenstate[qubit])
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.h(qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate.control(annotated="yes"))
        qc = qc.compose(qc_init)
        controlled_U = U.control(annotated="yes")
        qc.append(controlled_U, qargs = [qr_ancilla] + qr_eigenstate[:])
    
    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    print(qc)
    trans_qc = transpile(qc, backend, optimization_level=3)
    return trans_qc

def create_hamiltonian(parameters, scale=True, show_steps=False):
    '''
    Create a system hamiltonian for the Tranverse Field Ising Model

    Parameters:
     - parameters: a dictionary of parameters for contructing
       the Hamiltonian containing the following information
        - sites: the number of sites, default is 2
        - scaling: scales the eigenvalues to be in [-scaling, scaling]
        - shifting: shift the eigenvalues by this value
        - g: magnetic field strength
     - show_steps: if true then debugging print statements
                   are shown
    
    Effects:
       This method also creates parameter['r_scaling'] which
       is used for recovering the original energy.
     
    Returns:
     - H: the created hamiltonian
     - real_H_0: the minimum energy of the unscaled system
    '''
    scale_factor = parameters['scaling']
    shifting = parameters['shifting']
    if 'sites' in parameters.keys(): qubits = parameters['sites']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
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

# def create_hardware_backend():
#     '''
#     Creates a hardware backend using the inputted Qiskit user data.

#     Returns:
#      - backend: the specificed backend as a BackendV2 Qiskit Object
#     '''
#     hardware_name = input("Enter Hardware Backend Name:")
#     token    = input("Enter API Token:")
#     instance = input("Enter Instance:")
#     try:
#         print("Creating backend.")
#         service = QRS(channel='ibm_cloud', instance=instance, token=token)
#         backend = service.backend(hardware_name)
#         print("Backend created.")
#         return backend
#     except Exception as e:
#         print(e)
#         print("One or more of the provided service parameters are incorrect. Try again.")
#         create_hardware_backend()

def calculate_exp_vals(counts, shots):
    '''
    Calculates the real or imaginary of the expectation
    value depending on if the counts provided are from
    the real or the imaginary Hadamard tests.

    Parameters:
     - counts: the count object returned from result
     - shots: the number of shots used to run the tests with 

    Returns:
     - meas: the desired expection value
    '''
    p0 = 0
    if counts.get('0') is not None:
        p0 = counts['0']/shots
    meas = 2*p0-1
    return meas

if __name__ == '__main__':
    parameters = {}
    parameters['sites']    = 4
    parameters['scaling']  = 3*pi/4
    parameters['shifting'] = 0
    parameters['g']        = 1000 # magnetic field strength for TFIM

    H, E_0 = create_hamiltonian(parameters)
    
    eig_val, eig_vec = eigh(H)
    ground_state = eig_vec[:,0]/norm(eig_vec[:,0])
    print('ground', ground_state)
    statevector = [0]*(2**parameters['sites'])
    for i in range(len(ground_state)):
        # print(abs(ground_state[i])**2)
        if abs(ground_state[i]) > 1/len(ground_state):
            statevector[i] = 1
    statevector = statevector/norm(statevector)
    print('statevector', statevector)
    print('overlap', abs(statevector@ground_state.conj().T)**2)
    
    t = 1
    U = UnitaryGate(expm(-1j*H*t))
    
    shots = 100000

    use_hardware = True
    if use_hardware:
        backend = create_hardware_backend()
    else:
        backend = AerSimulator()
    
    sampler = Sampler(backend)
    trans_qcs = []
    # Real modified Hadamard test
    trans_qc = create_hadamard_tests(parameters, backend, U, modified=True)
    print('Real modified Hadamard test gate counts:', trans_qc.count_ops())
    trans_qcs.append(trans_qc)
    trans_qc = create_hadamard_tests(parameters, backend, U, W='Im', modified=True)
    print('Imaginary modified Hadamard test gate counts:', trans_qc.count_ops())
    trans_qcs.append(trans_qc)
    
    results = sampler.run(trans_qcs, shots = shots).result()
    
    raw_data = results[0].data
    cbit = list(raw_data.keys())[0]
    counts = raw_data[cbit].get_counts()
    Re = calculate_exp_vals(counts, shots)
    raw_data = results[1].data
    cbit = list(raw_data.keys())[0]
    counts = raw_data[cbit].get_counts()
    Im = calculate_exp_vals(counts, shots)

    print('Real:', eig_val[0])
    print('Estimate:', -np.log(complex(Re, Im)).imag)