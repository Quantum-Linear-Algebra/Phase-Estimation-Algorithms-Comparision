import numpy as np
from scipy.stats import truncnorm
from scipy.linalg import eigh
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService as QRS

from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit
import matplotlib.pyplot as plt

from scipy.linalg import expm

def create_hamiltonian(qubits, system, scale_factor, g=0, J=4, t=0, U=0, x=1, y=1, show_steps=False):
    assert(system[0:4].upper() == "TFIM" or system[0:4].upper() == "SPIN" or system[0:4].upper() == "HUBB" or system[0:4].upper() == "H2")
    # assert(abs(scale_factor)<=2*pi)
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if system[0:4].upper() == "TFIM":
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
    elif system[0:4].upper() == "SPIN":
        assert(J!=0)
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
    elif system[0:4].upper() == "HUBB":
        assert(x>=0 and y>=0)
        assert(x*y == qubits)

        # coupling portion
        Sd = np.array([[0,0],[1,0]])
        S = np.array([[0,1],[0,0]])
        I = np.eye(2)
        # op1 = np.kron(Sd, S)
        # op2 = np.kron(S, Sd)
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
                        # print(site_)
                        # print(site_ == site, site_ == neighbor)
                        if site_ == site: temp = np.kron(temp, op)
                        elif site_ == neighbor: temp = np.kron(temp, op.T)
                        else: temp = np.kron(temp, I) 
                    if temp.shape[0] == 64: print(temp)
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

    elif system[0:4].upper() == "H2":
        driver = PySCFDriver(
            atom=f'H .0 .0 .0; H .0 .0 {0.5}',
            unit=DistanceUnit.ANGSTROM,
            basis='sto3g'
        )

        molecule = driver.run()
        mapper = ParityMapper(num_particles=molecule.num_particles)
        hamiltonian = molecule.hamiltonian.second_q_op()
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        operator = tapered_mapper.map(hamiltonian)
        H = operator.to_matrix()
            
    if show_steps:
        val, vec = np.linalg.eigh(H)
        print("Original eigenvalues:", val)
        print("Original eigenvectors:\n", vec)
    
    # scale eigenvalues of the Hamiltonian
    if show_steps: print("Norm =", np.linalg.norm(H, ord=2))
    H = scale_factor*H/np.linalg.norm(H, ord=2)
    # rotate matrix so that it will be positive definite (not nessary in this usecase)
    # H += pi*np.eye(2**qubits)

    if show_steps:
        val, vec = np.linalg.eigh(H)
        print("Scaled eigenvalues:", val)
        print("Scaled eigenvectors:\n", vec)
        min_eigenvalue = np.min(val)
        print("Lowest energy eigenvalue", min_eigenvalue); print()
    
    return H


def create_HT_circuit(qubits, unitary, W = 'Re', backend = AerSimulator(), init_state = []):
    """
    Description: The code to create a Hadamard test circuits for a unitary operator 

    Args: number of qubits to represent the eigenstate: qubits; 
    time evolution unitary operator: unitary; 
    specifies real (imaginary) HT: W = 'Re'('Im'); 
    pecifies simulation (hardware) backend: backend = AerSimulator() (ibm_'hardware');
    eigenstate initialization with p0 overlap with ground_state: init_state

    Returns: a transpiled HT circuit: trans_qc
    """
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)

    qc.h(qr_ancilla)
    qc.initialize(init_state, qr_eigenstate[:])
    qc.append(unitary, qargs = [qr_ancilla[:]] + qr_eigenstate[:])

    # if W = Imaginary
    if W[0] == 'I': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])

    trans_qc = transpile(qc, backend, optimization_level=3)
    return trans_qc

def generate_Hadamard_test_data(ham,t_list,init_state,backend):
    """
    Input:
    ham: System Hamiltonian
    t_list: np.array of time points
   
    -Ouput:
    Z_Had: np.array of the output of Hadamard test (row)
    T_max: maximal Hamiltonian simulation time
    T_total: total Hamiltonian simulation time
    """
    shots = 1
    num_sites = np.log2(len(ham))
    Z_ests = []
    circs = []

    for t in t_list:
        mat = expm(-1j*ham*t)
        controlled_U = UnitaryGate(mat).control(annotated="yes")

        circs.append(create_HT_circuit(num_sites, controlled_U, W = 'Re', backend=backend, init_state=init_state))
        circs.append(create_HT_circuit(num_sites, controlled_U, W = 'Im', backend=backend, init_state=init_state))

    sampler = Sampler(backend)
    job = sampler.run(circs, shots = shots)
    results = job.result()

    for i in range(len(t_list)):

        re_data = results[2*i].data
        im_data = results[2*i+1].data

        re_counts = re_data[list(re_data.keys())[0]].get_counts()
        im_counts = im_data[list(im_data.keys())[0]].get_counts()

        if re_counts.get('0') is not None: Re = 1
        else: Re = -1

        if im_counts.get('0') is not None: Im = 1
        else: Im = -1

        Z_ests.append(complex(Re,Im))

    return np.array(Z_ests)

def generate_ts_distribution(T,N,sigma):
    """ 
    Generate time samples from truncated Gaussian
    Input:
    T : variance of Gaussian
    sigma : truncated parameter
    N : number of samples
    
    Output: 
    t_list: np.array of time points
    """
    t_list=truncnorm.rvs(-sigma, sigma, loc=0, scale=T, size=N)
    return t_list

def generate_Z(ham,T,N,sigma,init_state,backend):
    """ Generate Z samples for a given T,N,sigma
    Input:
    
    ham: System Hamiltonian
    T : variance of Gaussian
    N : number of time samples
    sigma : truncated parameter
    
    Output: 
    
    Z_est: np.array of Z output
    t_list: np.array of time points
    T_max: maximal running time
    T_total: total running time
    """
    t_list = generate_ts_distribution(T,N,sigma)
    Z_ests = generate_Hadamard_test_data(ham,t_list,init_state,backend)
    return Z_ests, t_list

def QMEGS_algo(Z_est, d_x, t_list, alpha, T):
    """ 
    Main routines for QMEGS
    Goal: Given signal, output estimatation of dominant frequencies
    -Input:
    Z_est: np.array of signal
    d_x: space step
    t_list: np.array of time points
    K: number of dominant frequencies
    alpha: interval constant
    T: maximal time

    -Output:
    Dominant_freq: np.array of estimation of dominant frequencies (up to adjustment when there is no gap)
    """
    num_x=int(2*np.pi/(d_x*10))
    num_x_detail=int(2*alpha/d_x/T)
    x_rough=np.arange(0,num_x)*d_x*10-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x_rough)))/len(Z_est)) #Gaussian filter function
    max_idx_rough = np.argmax(G)
    Dominant_potential=x_rough[max_idx_rough]
    x=np.arange(0,num_x_detail)*d_x+Dominant_potential-alpha/T
    G_detail=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est))
    max_idx_detail = np.argmax(G_detail)
    Dominant_freq=x[max_idx_detail]
    return Dominant_freq

def QMEGS_ground_energy(Z_ests,t_list,T_max,alpha,q, skipping = 1):
    """ 
    Uses QMEGS to estimate ground state energy
    -Input:
    Z_ests: np.array of signal
    t_list: np.array of time points
    T_max: maximal time
    alpha: interval constant
    q: searching parameter

    -Output:
    output_energy: ground state energy estimate
    len(Z_ests): number of observables
    T_total_QMEGS: Total running time of QMEGS
    """
    output_energies = []
    # T_totals = []
    for i in range(len(Z_ests)//skipping):
        idx = i*skipping
        d_x=q/T_max
        # T_totals.append(sum(np.abs(t_list[:i])))
        output_energies.append(QMEGS_algo(Z_ests[:idx], d_x, t_list[:idx], alpha, T_max))
        print('est', output_energies[i], 'real', eigs[0])
    plt.figure()
    plt.plot(abs(np.array(output_energies) - eigs[0]))
    plt.yscale('log')
    plt.savefig('QMEGS')
    return output_energies, [2*(i*skipping+1) for i in range(len(Z_ests)//skipping)]#, T_totals

if __name__ == '__main__':
    T = 1000 # maximal running time/ variance of gaussian
    N = 500
    sigma = 0.5
    q = 0.01
    alpha = q*100

    num_sites, system, scale_factor = 2, 'TFIM', 3*np.pi/4

    ham = create_hamiltonian(num_sites, system, scale_factor, g=4, J=1)
    eigs, vecs = eigh(ham)
    init_state = vecs[:,0]

    backend = AerSimulator()
    # api = input('token')
    # service = QRS(channel='ibm_cloud', instance='crn:v1:bluemix:public:quantum-computing:us-east:a/b8ff6077c08a4ea9871560ccb827d457:35e644dc-8b99-4215-957b-6ea07c220e44::', token=api)
    # backend = service.backend('ibm_rensselaer')

    Z_ests,t_list = generate_Z(ham,T,N,sigma,init_state,backend)
    energy_est, observables = QMEGS_ground_energy(Z_ests,t_list,T,alpha,q, skipping = 50)