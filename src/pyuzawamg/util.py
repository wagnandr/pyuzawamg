"""
Utility scripts and functions. 
"""
import numpy as np


def power_iteration(A, solver, verbose=False):
    """
    Calculates the spectral radius of 1 - solver * A.
    """
    u_new = A.create_vec()
    u_old = A.create_vec()
    u_new.randomize()
    u_old.randomize()
    solver.num_iterations = 1
    ev = np.nan
    rate = np.nan
    for i in range(40):
        # evaluate iteration matrix
        u_new = solver.solve(A * u_old)
        u_new = u_old - u_new
        # estimate eigenvalue
        n = u_old.inner(u_new)
        o = u_old.inner(u_old)
        ev_old, ev = ev, n/o
        # normalize
        u_new[:] /= u_new.norm()
        u_new, u_old = u_old, u_new
        rate_old, rate = rate, abs((ev_old-ev)/ev_old)
        if verbose:
            print(f'power iteration : {ev}, {rate}')
        if rate < 1e-2 and rate_old < 1e-2:
            break
    return ev


class CBCBlockWrapper:
    """
    Small wrapper to convert a block solver in our solver format.
    """
    def __init__(self, solver):
        self.solver = solver

    def mult(self, src, dst):
        #print(f'src , {src.norm()}')
        res = self.solver(initial_guess=dst) * src
        #print(f'res , {res.norm()}')
        dst[0][:] = res[0][:]