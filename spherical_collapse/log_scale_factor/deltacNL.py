import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.integrate import solve_ivp

def get_ics (x0, x2, cosmo, x_start=-10, x_end=0, atol=1e-8, feps=1e-22, rtol=1e-6, return_solution=False, dense_output=False, max_iter=10_000, method='RK45', ansatz='power-law'):

    a_start = np.exp(x_start)
    a_end = np.exp(x_end)
    # Auxiliary functions
    E = lambda a: cosmo.efunc(1/a-1)
    # Lets check if the derivative exists:
    try:
        cosmo.defunc(0.5)
        E_prime = lambda a: cosmo.defunc(1/a-1) * (-1/a**2)
    except:
        E_prime = nd.Derivative(E, n=1, step=atol, order=10)

    # Define the system of ODEs for equation (18)
    def odes(x, y):
        a = np.exp(x)
        delta, theta = y
        d_delta = theta
        d_theta = - (2 + a*E_prime(a)/E(a)) * theta + (4/3) * theta**2 / (1 + delta) + (3/2) * cosmo.Om(1/a-1) * delta * (1 + delta)
        return [d_delta, d_theta]

    def get_initial_ddot (x_start, delta0, ansatz='power-law'):

        if ansatz == 'power-law':

            a_start = np.exp(x_start)
            p = [1, 2*x_start + x_start * a_start * E_prime(a_start)/E(a_start) - 1, -3/2 * cosmo.Om(1/a_start-1) * x_start ** 2]
            r = np.roots(p)

            if np.any( np.iscomplex(r) ) or (np.min(r)>0):

                raise RuntimeError(f"I don't know what to do with {r}")

            return np.min(r) * delta0/x_start
        
        elif ansatz == 'exponential':

            a_start = np.exp(x_start)
            p = [1 - 4/3 * delta0/(1+delta0), 2 + a_start * E_prime(a_start)/E(a_start), -3/2 * cosmo.Om(1/a_start-1) * (1+delta0)]
            r = np.roots(p)

            alpha = np.max(r)

            if np.any( np.iscomplex(r) ) or (alpha<0):

                raise RuntimeError(f"I don't know what to do with {r}")

            return delta0 * alpha

        else:

            raise RuntimeError(f"Ansatz ({ansatz}) for determining theta not implemented!")

    def sol(delta0, dense_output=False, ansatz='power-law'):

        theta0 = get_initial_ddot(x_start, delta0, ansatz=ansatz) # initial velocity of perturbation
        y0 = [delta0, theta0]

        # Solve the ODE
        sol = solve_ivp(odes, (x_start, x_end), y0, method=method, atol=atol, rtol=rtol, dense_output=dense_output, t_eval=np.linspace(x_start, x_end, 10_000))
        return sol

    # Initial conditions
    f0 = sol(x0, ansatz=ansatz)
    f2 = sol(x2, ansatz=ansatz) 
    if not f0.success:
        raise RuntimeError("f0 does not converges")
    elif f2.success:
        raise RuntimeError("f2 does not diverges")
    for i in range(max_iter):

        x1 = (x0+x2)/2
        f1 = sol(x1, ansatz=ansatz) 

        if f1.success:

            x0 = x1

        else:

            x2 = x1

        if (np.abs(x2-x0) <= atol) and (1/f0.y[0, -1] <= feps):

            break

    if not return_solution:

        if f1.success:

            return x1

        else:

            return x0

    else:

        if f1.success:

            return (x1, sol(x1, dense_output=dense_output, ansatz=ansatz), sol(x2, dense_output=dense_output, ansatz=ansatz)) if dense_output else (f1, f2)

        else:

            return (x0, sol(x0, dense_output=dense_output, ansatz=ansatz), sol(x1, dense_output=dense_output, ansatz=ansatz)) if dense_output else (f0, f1)
       
