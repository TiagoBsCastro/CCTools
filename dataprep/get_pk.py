# cctools/get_pk.py
import numpy as np
from mpi4py import MPI
from cctools.cctools import get_n_samples, get_sigma_r
from cctools.cctools import generate_power_spectra as generate_power_spectra

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the parameter ranges: [min, max]
param_ranges = {
    'h': [0.6, 0.8],
    'Omega_b': [0.03, 0.06],
    'Omega_c': [0.1, 0.45],
    'A_s': [1.5e-9, 2.5e-9],
    'n_s': [0.9, 1.0],
    'mnu': [0.0, 0.2],       # Neutrino mass
    'Omega_Lambda': [0.6, 0.75], # Omega_Lambda
    'redshift': [0.0, 3.0]   # Redshift
}
parameters = list(param_ranges.keys())

# Number of samples
n_samples = 1_000_000
# Number of Modes
R_values = np.logspace(-1, 1.5, 50)

if n_samples % size:

    raise RuntimeError(f"n_samples ({n_samples}) is not divisible for MPI size {size}")

if rank == 0:
    # Master process generates the samples
    params_array = get_n_samples(param_ranges, n_samples)
    params_splits = np.array_split(params_array, size, axis=0)

    # Distribute the data to other processes
    for i in range(1, size):
        comm.send(params_splits[i], dest=i, tag=i)
    # Master gets its own portion
    params_array = params_splits[0]
else:
    # Worker processes receive their portion
    params_array = comm.recv(source=0, tag=rank)

# The params_array is now split among the processes
training_parameters = {}
for i, k in enumerate(parameters):

    training_parameters[k] = params_array[:, i]

# Compute power spectra in each process
power_spectra = generate_power_spectra(params_array)
sigma_R_array = get_sigma_r(power_spectra, R_values)

# Gather power spectra back to the master process
gathered_power_spectra = comm.gather(power_spectra, root=0)
gathered_sigma_R_array = comm.gather(sigma_R_array, root=0)

if rank == 0:
    # Flatten the list of gathered power spectra
    all_power_spectra = np.array([item for sublist in gathered_power_spectra for item in sublist])
    all_sigma_R_array = np.array([item for sublist in gathered_sigma_R_array for item in sublist])
    np.savez("data.npz", sigma_R_array=all_sigma_R_array, power_spectra=all_power_spectra)
else:
    pass
