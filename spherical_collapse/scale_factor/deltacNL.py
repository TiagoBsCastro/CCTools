import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.integrate import solve_ivp

def get_ics (x0, x2, cosmo, a_start=1e-3, a_end=1.0, atol=1e-8, rtol=1e-6, return_solution=False, dense_output=False, max_iter=100, method='RK45'):

    # Auxiliary functions
    E = lambda a: cosmo.efunc(1/a-1)
    # Lets check if the derivative exists:
    try:
        cosmo.defunc(0.5)
        E_prime = lambda a: cosmo.defunc(1/a-1) * (-1/a**2)
    except:
        E_prime = nd.Derivative(E, n=1, step=atol, order=10)

    # Define the system of ODEs for equation (18)
    def odes(a, y):
        delta, theta = y
        d_delta = theta
        d_theta = - (3/a + E_prime(a)/E(a)) * theta + (4/3) * theta**2 / (1 + delta) + (3/2) * (cosmo.Om0 / (a**5 * E(a)**2)) * delta * (1 + delta)
        return [d_delta, d_theta]

    def get_initial_ddot (a_start, delta0):

        p = [a_start**3 * (1-4*delta0/3/(1 + delta0)), 2*a_start**3 + E_prime(a_start)/E(a_start)*a_start**4, -3/2 * cosmo.Om0/E(a_start)**2 * (1+delta0)]
        r = np.roots(p)

        if ( (r[0]<0) or np.iscomplex(r[0]) ) and ( (r[1]>0) and np.isreal(r[1]) ):

            return r[1]

        else:

            raise RuntimeError(f"I dont know what to do with solution: {r}")

    def sol(delta0, dense_output=False):

        theta0 = delta0*get_initial_ddot(a_start, delta0)/a_start # initial velocity of perturbation
        y0 = [delta0, theta0]

        # Solve the ODE
        sol = solve_ivp(odes, (a_start, a_end), y0, method=method, atol=atol, rtol=rtol, dense_output=dense_output)
        return sol

    # Initial conditions
    f0 = sol(x0)
    f2 = sol(x2)
    if not f0.success:
        raise RuntimeError("f0 does not converges")
    elif f2.success:
        raise RuntimeError("f2 does not diverges")
    for i in range(max_iter):

        x1 = 2*(x0*x2)/(x0+x2)
        f1 = sol(x1)

        if f1.success:

            x0 = x1

        else:

            x2 = x1

        if (np.abs(x2-x0) <= atol) and (1/f0.y[0, -1] <= atol):

            break

    if not return_solution:

        if f1.success:

            return x1

        else:

            return x0

    else:

        if f1.success:

            return x1, sol(x1, dense_output=dense_output) if dense_output else f1

        else:

            return x0, sol(x0, dense_output=dense_output) if dense_output else f0
       
