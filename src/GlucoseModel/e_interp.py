# Returns the value of e off the "time grid" of k*Delta (Delta = 15mins) using cubic spline interpolation.
import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

def e_interp(e_disc,t_step,t):
    n = len(e_disc)
    print(n)
    ts = np.arange(0.0,float(n),1.0) * t_step
    print(ts)
    e_cont = CubicSpline(ts,e_disc,bc_type='natural')
    return e_cont(t)
