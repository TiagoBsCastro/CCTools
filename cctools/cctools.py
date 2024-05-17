# cctools/cctools.py
import numpy as np
import baccoemu
import camb
from camb import model, initialpower
from pyDOE import lhs
from multiprocessing import cpu_count

# Set the number of threads
num_threads = cpu_count()

def get_n_samples(param_ranges, n_samples):

    # Generate Latin hypercube samples
    lhs_samples = lhs(len(param_ranges), samples=n_samples)

    # Scale samples to the parameter ranges
    params_list = []
    for i, key in enumerate(param_ranges):
        min_val, max_val = param_ranges[key]
        scaled_samples = lhs_samples[:, i] * (max_val - min_val) + min_val
        params_list.append(scaled_samples)

    return np.array(params_list).T

def generate_power_spectra(params_list, var=7):
    power_spectra = []
    for params in params_list:
        h, Omega_b, Omega_c, A_s, n_s, mnu, Omega_Lambda, redshift = params
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, mnu=mnu, omk=(1-Omega_Lambda-Omega_b-Omega_c))
        pars.InitPower.set_params(As=A_s, ns=n_s)
        pars.set_matter_power(redshifts=[redshift], kmax=10.0)
        pars.NumThreads = num_threads

        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=200, var1=var, var2=var)

        power_spectra.append((kh, pk[0]))
    return power_spectra

def generate_power_spectra_bacco(params_list):
    power_spectra = []
    bacco = baccoemu.Matter_powerspectrum()
    for params in params_list:
        h, Omega_b, Omega_c, A_s, n_s, mnu, Omega_Lambda, redshift = params
        k = np.logspace(-3, np.log10(5), num=100)
        params = {
            'omega_cold'    :  Omega_c + Omega_b,
            'A_s'           :  A_s, # if A_s is not specified
            'omega_baryon'  :  Omega_b,
            'ns'            :  n_s,
            'hubble'        :  h,
            'neutrino_mass' :  mnu,
            'w0'            : -1.0,
            'wa'            :  0.0,
            'expfactor'     :  1/(redshift + 1)
        }
        kh, pk = bacco.get_linear_pk(k=k, cold=False, **params)
        power_spectra.append((kh, pk))
    return power_spectra

def compute_sigma_r(k, pk, R):

    R = np.atleast_1d(R)
    if R.size > 1:
        R = R[:,np.newaxis]
        axis = 1
    else:
        axis = 0
    Wk = 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3
    sigma_r_squared = np.trapz(pk * Wk**2 * k**2, x=k, axis=axis) / (2 * np.pi**2)
    return np.sqrt(sigma_r_squared)

def get_sigma_r (power_spectra, R_values):

    # Compute sigma(R) for a range of R values
    sigma_R_data = []

    for kh, pk in power_spectra:
        sigma_R = compute_sigma_r(kh, pk, R_values)
        sigma_R_data.append(sigma_R)

    return np.array(sigma_R_data)
