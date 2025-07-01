import pickle
from qiskit_ibm_runtime import QiskitRuntimeService as QRS
from os import remove
from os.path import getsize
from sys import exit

def empty(filename):
    try:
        return getsize(filename) == 0
    except FileNotFoundError:
        return True

def create_hardware_backend():
    '''
    Creates a hardware backend using the inputted Qiskit user data.

    Returns:
     - backend: the specificed backend as a BackendV2 Qiskit Object
    '''
    filename = "Service.pkl" 
    if empty(filename):
        token    = input("Enter API Token:")
        instance = input("Enter Instance:")
        hardware_name = input("Enter Hardware Backend Name:")
        with open(filename, 'wb') as file:
            pickle.dump([token, instance, hardware_name], file)
    else:
        with open(filename, 'rb') as file:
            [token, instance, hardware_name] = pickle.load(file)
    try:
        print("Creating backend.")
        service = QRS(channel='ibm_cloud', instance=instance, token=token)
        backend = service.backend(hardware_name)
        print("Backend created.")
        return backend
    except Exception as e:
        print(e)
        print("One or more of the provided service parameters are incorrect. Try rechecking your IBM Quantum Platform.")
        if not empty(filename): remove(filename)
        exit(1)

    
def create_service():
    '''
    Creates a hardware backend using the inputted Qiskit user data.

    Returns:
     - service: the specificed service as a Qiskit Service Object
    '''
    filename = "Service.pkl" 
    if empty(filename):
        token    = input("Enter API Token:")
        instance = input("Enter Instance:")
        hardware_name = input("Enter Hardware Backend Name:")
        with open(filename, 'wb') as file:
            pickle.dump([token, instance, hardware_name], file)
    else:
        with open(filename, 'rb') as file:
            [token, instance, hardware_name] = pickle.load(file)
    try:
        print("Creating Service.")
        service = QRS(channel='ibm_cloud', instance=instance, token=token)
        print("Service saved.")
        return service
    except Exception as e:
        print(e)
        print("One or more of the provided service parameters are incorrect. Try rechecking your IBM Quantum Platform.")
        if not empty(filename): remove(filename)
        exit()

if __name__ == '__main__':
    create_service()