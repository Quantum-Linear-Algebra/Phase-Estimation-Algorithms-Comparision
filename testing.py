import sys 
paths = ['./0-Data', './1-Algorithms', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
import Data_Manager as data
from Service import create_hardware_backend

from numpy import pi, log, array
import numpy as np
from scipy.linalg import eigh, norm

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation
from qiskit_aer import AerSimulator
from qiskit import transpile

def create_hadamard_tests(backend, U, statevector, W = 'Re', test=False):
    '''
    Creates a transpiled hadamard tests for the specificed backend.

    Parameters:
     - backend: the backend to transpile the circuit on
     - controlled_U: the control operation to check phase of
     - statevector: a vector to initalize the statevector of
                    eigenqubits
     - W: what type of hadamard tests to use (Re or Im)
    
    Returns:
     - trans_qc: the transpiled circuit
    '''
    if test:
        zero_sv = array([1]+[0]*(len(U)-1))
        ev = complex(zero_sv.conj().T@U@zero_sv)
        phase_shift = log(ev).imag
    
    U = UnitaryGate(U)
    if not test: controlled_U = U.control(annotated="yes")
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(U.num_qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)

    if not test:
        qc.initialize(statevector, qr_eigenstate)
        qc.append(controlled_U, qargs = [qr_ancilla[:]] + qr_eigenstate[:])
    else:
        if parameters['g'] > 1:
            for i in range(parameters['sites']): qc.ch(0,i+1)
        else:
            qc.ch(0,1)
            for i in range(parameters['sites']-1): qc.cx(1,i+2)

        # state_prep = StatePreparation(statevector)
        # gate = state_prep.control(annotated="yes")
        # qc.append(gate, qargs=[qr_ancilla]+qr_eigenstate[:])
        qc.append(U, qargs = qr_eigenstate)
        qc.x(qr_ancilla)
        # qc.append(gate, qargs=[qr_ancilla]+qr_eigenstate[:])
        
        if parameters['g'] > 1:
            for i in range(parameters['sites']): qc.ch(0,i+1)
        else:
            qc.ch(0,1)
            for i in range(parameters['sites']-1): qc.cx(1,i+2)

        qc.x(qr_ancilla)
        qc.rz(phase_shift, qr_ancilla)

    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    print(qc)
    
    gates = ['ecr', 'sx', 'x', 'rz', 'id']
    trans_qc = transpile(qc, backend, basis_gates=gates, optimization_level=3)
    print(trans_qc)
    return trans_qc


if __name__ == '__main__':
    parameters = {}
    # SPECIFIC SYSTEM TYPE
    parameters['sites']        = 4
    parameters['scaling']      = 3*pi/4
    parameters['shifting']     = 0
    parameters['system']     = 'TFI' # OPTIONS: TFIM, SPIN, HUBBARD, H_2

    # Transverse Field Ising Model Parameters
    parameters['g'] = .1 # magnetic field strength (TFIM)
    parameters['method_for_model'] = 'Q' # OPTIONS: F3C, Qiskit
    parameters['trotter'] = 10 # only with method_for_model = F3C

    shots = 10000
    
    H, E_0 = data.create_hamiltonian(parameters)

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
    U = data.closest_exp_unitary(H,t)

    hardware = False
    if hardware:
        backend = AerSimulator()
        # backend = create_hardware_backend()
    else:
        backend = AerSimulator()

    trans_qc = create_hadamard_tests(backend, U, statevector)
    two_gates_count = 0
    for instruction in trans_qc.data:
        if instruction.operation.num_qubits == 2:
            two_gates_count+=1
    print('Regular Hadamard Test two qubit gate count re', two_gates_count)
    
    if not hardware:
        counts = backend.run(trans_qc, shots = shots).result().get_counts()
        Re = data.calculate_exp_vals(counts, shots)
    trans_qc = create_hadamard_tests(backend, U, statevector, W='Im')

    two_gates_count = 0
    for instruction in trans_qc.data:
        if instruction.operation.num_qubits == 2:
            two_gates_count+=1
    print('Regular Hadamard Test two qubit gate count im', two_gates_count)
    
    if not hardware:
        counts = backend.run(trans_qc, shots = shots).result().get_counts()
        Im = data.calculate_exp_vals(counts, shots)
    
    if not hardware:
        print(eig_val[0], -log(complex(Re, Im)).imag)

    trans_qc = create_hadamard_tests(backend, U, statevector, test=True)
    two_gates_count = 0
    for instruction in trans_qc.data:
        if instruction.operation.num_qubits == 2:
            two_gates_count+=1
    print('Modified Hadamard Test two qubit gate count re', two_gates_count)
    
    if not hardware:
        counts = backend.run(trans_qc, shots = shots).result().get_counts()
        Re = data.calculate_exp_vals(counts, shots)
    trans_qc = create_hadamard_tests(backend, U, statevector, W='Im', test=True)

    two_gates_count = 0
    for instruction in trans_qc.data:
        if instruction.operation.num_qubits == 2:
            two_gates_count+=1
    print('Modified Hadamard Test two qubit gate count im', two_gates_count)

    if not hardware:
        counts = backend.run(trans_qc, shots = shots).result().get_counts()
        Im = data.calculate_exp_vals(counts, shots)
    
    if not hardware:
        print(eig_val[0], -log(complex(Re, Im)).imag)