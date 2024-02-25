import sys
# Add the custom library directory to Python's search path for modules.
sys.path.append('/Users/ethanelliotrajkumar/Documents/DVR-VQE2.0/lib')
# Import modules and functions from the custom library.
from greedy_circs import *  # Assuming functions for circuit optimization
import numpy as np  # Importing NumPy for numerical computations
from dvr1d import *  # For 1D Discrete Variable Representation operations
from utils import *  # Utility functions that might be widely used
from vqe import DVR_VQE  # Importing the class for DVR using VQE
from pot_gen import get_pot_cr2  # Function to generate potential for CR2

# 16 points
params32 = [2.8, 4]

dvr_options = {
    'type': '1d',
    'box_lims': (params32[0], params32[1]),
    'dx': (params32[1] - params32[0]) / 32,
    'count': 32
}

ansatz_options_list = [{
    'type': 'greedy',
    'constructive': True,
    'layers': 6 * reps,
    'num_keep': 1,
    'add_h': False, 
    'add_sx': True,
    'add_rs': True,
    'samples': 20, 
    'num_qubits': 5,
    'reps': reps,
    'partitions': [[0,1,2,3,4]],
    'max_gate': 10 
    } for reps in [1, 2, 3]]

vqe_options = {
    'optimizers':['NFT.100'],
    'repeats': 5, 
    'seed': 42
}


mol_params = cr2_params
spin = 1
mol_params['name'] += f'_{spin}'


pot, lims = get_pot_cr2(spin)

dvr_vqe = DVR_VQE(mol_params, pot, log_dir='/Users/ethanelliotrajkumar/Documents/DVR-VQE2.0/results/cr2/spin_state_1/greedy/5_qubits')
# dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'test/')
for ansatz_options in ansatz_options_list:
    dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=False)
    