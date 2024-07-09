import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import sys
import astropy.units as u
from astropy.cosmology import Flatw0waCDM
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
from scipy.interpolate import CubicSpline
from deltacNL import get_ics

####################### input ##########################

if len(sys.argv) == 3:

    class Cosmology:
        def __init__ (self, bkgfile):

            bkg = np.loadtxt(bkgfile)
            self.z = bkg[::-1, 0]
            self.ez = bkg[::-1, 3]
            self.ez /= self.ez[0]
            # Redshift derivative
            self.efunc = CubicSpline(self.z, self.ez)
            self.defunc = self.efunc.derivative(1)
            # Om and Ode
            Om  = (bkg[::-1, 10]+bkg[::-1, 9])/bkg[::-1, 14]
            Ode = bkg[::-1, 11]/bkg[::-1, 14]
            self.Om  = CubicSpline(self.z, Om)
            self.Ode = CubicSpline(self.z, Ode)
            self.Om0 = self.Om(0)
            self.Ode0 = self.Ode(0) 

    cosmo = Cosmology(sys.argv[1])
    fout = sys.argv[2]

else:

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
a_start_factor = 1e-5
a_sup          = 1.0
a_inf          = 1/6
N_a            = 10
atol           = 1e-10
rtol           = atol*100
feps           = 1/atol # Minimum value of 1/deltaNL(z)
delta_inf      = 1e-6   # First value for the bissection root finder
delta_sup      = 1e0    # Second value for the bissection root finder
method_linear  = 'DOP853' 
method_nl      = 'DOP853' 
ansatz         = 'exponential'
plot           = False
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
# Lets check if the derivative exists:
try:
    cosmo.defunc(0.5)
    E_prime = lambda a: cosmo.defunc(1/a-1) * (-1/a**2)
except:
    E_prime = nd.Derivative(E, n=1, step=atol, order=10)

def get_DeltaV (cosmo, nlsol, ac, atol=atol):

    rinv = lambda a: ( np.abs(1+nlsol(a)[0]) )**(1/3) / a
    eds_guess = ac / (2**(2/3))
    minsol = minimize(rinv, eds_guess, tol=atol, bounds=[(.1*eds_guess, 0.99*ac)])
    amax = minsol.x[0]
    rmax = 1/minsol.fun
    Dta  = nlsol(amax)[0] + 1
    def yvir (a):

        etat = 2/Dta * cosmo.Ode(1/amax-1)/cosmo.Om(1/amax-1)
        etav = 2/Dta * (amax/a) ** 3 * cosmo.Ode(1/a-1)/cosmo.Om(1/a-1)
        return (1 - etav/2)/(2+etat-3/2*etav)

    # DeltaV @ collapse
    DeltaVc = Dta * (ac/amax/yvir(ac))**3

    # Solving for DeltaV @ virialization
    av = fsolve( lambda _: _/amax * (Dta/(1+nlsol(_)[0]))**(1/3) - yvir(_), (amax+ac)/2)[0]
    DeltaVv = Dta * (av/amax/yvir(av))**3
    
    return DeltaVc, DeltaVv, av, amax

# Define the system of ODEs for equation (19)
def odes(x, y):
    a = np.exp(x)
    delta, theta = y
    d_delta = theta
    d_theta = - (2 + a*E_prime(a)/E(a)) * theta + (3/2) * cosmo.Om(1/a-1) * delta
    return [d_delta, d_theta]

def main(plot=False):

    results = []

    for a_end in np.linspace(a_inf, a_sup, N_a):

        a_start = a_start_factor * a_end

        ############### Initial conditions #####################

        x_start = np.log(a_start)
        x_end = np.log(a_end)

        _, nlsol, divsol = get_ics (delta_inf, delta_sup, cosmo, x_start=x_start, x_end=x_end, ansatz=ansatz, feps=feps,
                                    atol=atol, rtol=rtol, return_solution=True, dense_output=True, method=method_nl)

        _delta0 = divsol.sol(x_start)[0]
        p = [1, 2 + a_start * E_prime(a_start)/E(a_start), -3/2 * cosmo.Om(1/a_start-1)]
        r = np.max(np.roots(p))
        
        y0 = [_delta0, _delta0*r]

        ################## Solve the ODE #######################

        sol = solve_ivp(odes, (x_start, x_end), y0, method=method_linear, atol=atol, rtol=rtol, dense_output=True, t_eval=np.linspace(x_start, x_end, 10_000))

        # Extract the solution
        delta_sol = lambda a: sol.sol(np.log(a))
        delta_sol_nl = lambda a: nlsol.sol(np.log(a))

        if plot:
            # Plotting the solution
            a = np.geomspace(a_start, a_end, 1000)
            plt.loglog(a, delta_sol(a)[0], label="linear")
            plt.loglog(a, delta_sol(a)[1], label="linear derivative")
            plt.loglog(a, delta_sol_nl(a)[0], label="non-linear")
            plt.loglog(a, delta_sol_nl(a)[1], label="non-linear derivative")
            plt.xlabel('Scale factor a')
            plt.ylabel('Density perturbation δ')
            plt.title('Evolution of density perturbation δ')
            plt.grid(True)
            plt.legend()
            plt.ylim(a_start, 1e3)
            plt.show()

        # Delta Virial
        DeltaVc, DeltaVv, av, amax = get_DeltaV (cosmo, delta_sol_nl, a_end, atol=atol)

        results += [[a_end, amax, av, delta_sol(amax)[0], delta_sol(av)[0], delta_sol(a_end)[0], DeltaVv, DeltaVc, delta_sol_nl(a_end)[0]]]

        a = np.geomspace(a_start, a_end, 1000)
        np.savez(f"{fout}_spherical_colapse_solution_a={a_end:5.4f}", a=a, lin=delta_sol(a), nl=delta_sol_nl(a))

    np.savetxt(fout+".txt", results)

main(plot=plot)
