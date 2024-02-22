from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2
from lib.dvr1d import *
from lib.utils import *
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
import numpy as np 
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit_ibm_runtime import Options 
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator as RuntimeEstimator

def run_VQE_for_all_resilience_levels(mol_params, spin, params16, backends, token):
    # service = QiskitRuntimeService(channel='ibm_quantum', token=token)
     # service = QiskitRuntimeService(channel='ibm_quantum', token='4c079342f5096081e451eed8b8acfee35dc5482e87877b470a91bcd43d6d7d9afe9ece12dbb4e738b2bc4e4cc9279db76fa53ad57196d43cb5d8c0f731d17c9a')

    def gen_ansatz_op(mol_params, spin, params16):
        mol_params = mol_params.copy()  # create a copy to avoid changing the original mol_params
        mol_params['name'] += f'_{spin}'
        dvr_options = {
            'type': '1d',
            'box_lims': (params16[0], params16[1]),
            'dx': (params16[1] - params16[0]) / 16,
            'count': 16
        }

        # obtain the potential for a CR2 at certain radii
        pot, lims = get_pot_cr2(spin)

        # perform a dvr vqe to obtain the hamiltonian
        dvr_vqe = DVR_VQE(mol_params, pot)
        h_dvr = dvr_vqe.get_h_dvr(dvr_options, J=0) * hartree

        # Perform a Pauli Decomposition to get the Hamiltonian and get a composition.
        h_dvr_p0 = SparsePauliOp.from_operator(h_dvr)
        print(h_dvr_p0.coeffs)

        num_qubits = int(np.log2(h_dvr.shape[0]))
        a = TwoLocal(num_qubits, rotation_blocks=['ry'], entanglement_blocks=['cx'], entanglement='linear', reps=3).decompose()
        
        return h_dvr, h_dvr_p0, a
    
    def Run_VQE(a, optimizers, params, estimator, operator):
        """ This is the function that runs the VQE."""
        repeat = 3
        params = None
        # params = np.array([float(i) for i in result.optimal_parameters.values()])
        converge_cnts1 = np.empty([len(optimizers)], dtype=object)
        converge_vals1 = np.empty([len(optimizers)], dtype=object)
        converge_params = np.empty([len(a.params)], dtype=object)
        for i, optimizer in enumerate(optimizers):
            print('Optimizer: {}        '.format(type(optimizer).__name__))
            # algorithm_globals.random_seed = 42

            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                parameters = np.array(parameters)
                print(f'\r{eval_count}, {mean}, {parameters}', end='')
            best_res1 = None
            
            for j in range(repeat):
                counts = []
                values = []
                parameters = []
                vqe = VQE(estimator=estimator, ansatz=a, optimizer=optimizer, initial_point=params, callback=store_intermediate_result)
                results = vqe.compute_minimum_eigenvalue(operator=operator)
                print()
                if (best_res1 is None) or (values[-1] <= best_res1):
                    best_res1 = values[-1]
                    converge_cnts1[i] = np.asarray(counts)
                    converge_vals1[i] = np.asarray(values)
                    converge_params[i] = np.asarray(parameters) 
                    
        print('\nOptimization complete ') 
        return converge_cnts1, converge_vals1, converge_params, results


    h_dvr, operator, a = gen_ansatz_op(mol_params, spin, params16)
    service = QiskitRuntimeService(channel='ibm_quantum', token=token)
    shots = 8192
    params = None
    options = Options(max_execution_time=180000)
    optimizers = [COBYLA(maxiter=100), SPSA(maxiter=200)] 
    options.execution.shots = shots
    converge_cnts_list = []
    converge_vals_list = []
    results_list = []
    for i in range(4):  # Assuming resilience levels from 0 to 3
        print(f"Running for resilience level {i}...")
        options.resilience_level = i
        with Session(service=service, backend=backends) as session: 
            estimator = RuntimeEstimator(session=session)
            converge_cnts, converge_vals, result = Run_VQE(a, optimizers, params, estimator, operator)
            converge_cnts_list.append(converge_cnts)
            converge_vals_list.append(converge_vals)
            results_list.append(result)
    return converge_cnts_list, converge_vals_list, results_list, h_dvr

def dvr_VQE(mol_params, spins, params16_list, backends, token):
    # Initialize empty lists to store results for each spin
    all_converge_cnts_list = []
    all_converge_vals_list = []
    all_results_list = []
    all_h_dvr_list = []
    
    # Loop over all spins and corresponding molecular parameters
    for i, (params16, spin) in enumerate(zip(params16_list, spins)):
        print(f"Running for spin {i}...")
        converge_cnts_list, converge_vals_list, results_list, h_dvr = run_VQE_for_all_resilience_levels(mol_params, spin, params16, backends, token)

        # Append the results for this spin to the overall results lists
        all_converge_cnts_list.append(converge_cnts_list)
        all_converge_vals_list.append(converge_vals_list)
        all_results_list.append(results_list)
        all_h_dvr_list.append(h_dvr)
        
    return all_converge_cnts_list, all_converge_vals_list, all_results_list, all_h_dvr_list

def convert_to_dict(all_converge_cnts_list, all_converge_vals_list, all_results_list, all_h_dvr_list):
    result_dict = {}
    for i, (converge_cnts_list, converge_vals_list, results_list, h_dvr) in enumerate(zip(all_converge_cnts_list, all_converge_vals_list, all_results_list, all_h_dvr_list)):
        for resilience_level in range(4):
            key_base = f"s_{i}_r_{resilience_level}_"
            result_dict[key_base + "converge_cnts"] = converge_cnts_list[resilience_level]
            result_dict[key_base + "converge_vals"] = converge_vals_list[resilience_level]
            result_dict[key_base + "result"] = results_list[resilience_level]
            result_dict[key_base + "h_dvr"] = h_dvr
    return result_dict
