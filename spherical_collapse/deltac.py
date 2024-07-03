import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import sys
import astropy.units as u
from astropy.cosmology import Flatw0waCDM
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
from deltacNL import get_ics

####################### input ##########################

Om0   = float(sys.argv[1])
Ob0   = 0 if Om0 == 1 else float(sys.argv[2])
m_nu  = 0 if Om0 == 1 else float(sys.argv[3])
H0    = sys.argv[4] 
ns    = sys.argv[5] # Doesnt matter
As    = sys.argv[6] # Doesnt matter
w0    = 0 if Om0 == 1 else float(sys.argv[7])
wa    = 0 if Om0 == 1 else float(sys.argv[8])
fout  = sys.argv[9]

## Other things to set 
Tcmb0 = 0 if Om0 == 1 else 2.7255
Neff  = 0 if Om0 == 1 else 3.046

# on concept I considered Om0 to contain neutrinos
if m_nu > 1e-4:
    aux = Flatw0waCDM(H0=H0, Om0=Om0, m_nu=[m_nu/3*u.eV]*3, Ob0=Ob0, w0=w0, wa=wa, Tcmb0=Tcmb0*u.K, Neff=Neff)
    Om0 = Om0 - aux.Onu0
cosmo = Flatw0waCDM(H0=H0, Om0=Om0, m_nu=[m_nu/3*u.eV]*3, Ob0=Ob0, w0=w0, wa=wa, Tcmb0=Tcmb0*u.K, Neff=Neff)

# Techinical parameters
a_sup     = 1.0
a_inf     = 0.1
N_a       = 2
atol      = 1e-8
rtol      = atol*100
delta_inf = 1e-5 # First value for the bissection root finder
delta_sup = 1e-1 # Second value for the bissection root finder

############# Auxiliary solutions @ EdS ################

eds_sol = {
        'delta_ta' : 3/5 * (3*np.pi/4)**(2/3),
        'delta_c'  : 3/20 * (12*np.pi)**(2/3),
        'delta_v'  : 3/20 * (6 + 9*np.pi)**(2/3),
        'DeltaVc'  : 18*np.pi**2,
        'DeltaVv'  : 18*np.pi**2 * ( 3/4 + 1/2/np.pi )**2
        }

################ Auxiliary functions ###################

E = lambda a: cosmo.efunc(1/a-1)
E_prime = nd.Derivative(E, n=1, step=atol, order=5)

def get_DeltaV (cosmo, nlsol, ac, atol=atol):

    rinv = lambda a: ( np.abs(1+nlsol.sol(a)[0]) )**(1/3) / a
    eds_guess = ac / (2**(2/3))
    minsol = minimize(rinv, eds_guess, tol=atol, bounds=[(.1*eds_guess, 0.99*ac)])
    amax = minsol.x[0]
    rmax = 1/minsol.fun
    Dta  = nlsol.sol(amax)[0] + 1
    def yvir (a):

        etat = 2/Dta * cosmo.Ode(1/amax-1)/cosmo.Om(1/amax-1)
        etav = 2/Dta * (amax/a) ** 3 * cosmo.Ode(1/a-1)/cosmo.Om(1/a-1)
        return (1 - etav/2)/(2+etat-3/2*etav)

    # DeltaV @ collapse
    DeltaVc = Dta * (ac/amax/yvir(ac))**3

    # Solving for DeltaV @ virialization
    av = fsolve( lambda _: _/amax * (Dta/(1+nlsol.sol(_)[0]))**(1/3) - yvir(_), (amax+ac)/2)[0]
    DeltaVv = Dta * (av/amax/yvir(av))**3
    
    return DeltaVc, DeltaVv, av, amax

# Define the system of ODEs for equation (19)
def odes(a, y):
    delta, theta = y
    d_delta = theta
    d_theta = - (3/a + E_prime(a)/E(a)) * theta + (3/2) * (cosmo.Om0 / (a**5 * E(a)**2)) * delta
    return [d_delta, d_theta]

def main(plot=False):

    results = []

    for a_end in np.linspace(a_inf, a_sup, N_a):

        a_start = 1e-5 * a_end

        ############### Initial conditions #####################

        delta0, (nlsol, theta0) = get_ics (delta_inf, delta_sup, cosmo, a_start=a_start, a_end=a_end, 
                                           atol=atol, rtol=rtol, return_solution=True, dense_output=True)   # small initial perturbation
        y0 = [delta0, theta0]

        ################## Solve the ODE #######################

        sol = solve_ivp(odes, (a_start, a_end), y0, method='RK45', atol=atol, rtol=rtol, dense_output=True)

        # Extract the solution
        delta_sol = sol.sol

        if plot:
            # Plotting the solution
            a = np.geomspace(a_start, a_end, 1000)
            plt.loglog(a, delta_sol(a)[0])
            plt.loglog(a, nlsol.sol(a)[0])
            plt.xlabel('Scale factor a')
            plt.ylabel('Density perturbation δ')
            plt.title('Evolution of density perturbation δ')
            plt.grid(True)
            plt.ylim(a_start, 1e3)
            plt.show()

        # Delta Virial
        DeltaVc, DeltaVv, av, amax = get_DeltaV (cosmo, nlsol, a_end, atol=atol)

        results += [[a_end, amax, av, delta_sol(amax)[0], delta_sol(av)[0], delta_sol(a_end)[0], DeltaVv, DeltaVc]]

    np.savetxt(fout+".txt", results)

main()
