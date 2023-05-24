# Same as "e_interp" but for u. Is it actually used anywhere?
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def u_interp(u_disc,t_step,t):
    n = len(u_disc)
    print(n)
    ts = np.arange(0.0,float(n),1.0) * t_step
    print(ts)
    u_cont = CubicSpline(ts,u_disc,bc_type='natural')

    # Should it be u_cont?
    return u_cont(t)
